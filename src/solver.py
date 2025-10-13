import os
import time
import logging
import nbformat
from pathlib import Path
from typing import Optional, List, Dict, Any

from client import GeminiClient
from parser import NotebookTaskSplitter
from runner import CodeRunner, ExecutionResult
from executor import LastCellExecutor, ICodeFixer
from prompter import TaskPrompter
from history import ConversationHistory


class GeminiFixer:
    """
    Фиксер ошибок кода через Gemini.
    Работает В РАМКАХ открытой ветки истории — добавляет запрос на исправление и ответ модели.
    """

    def __init__(
        self,
        gemini: GeminiClient,
        history: ConversationHistory,
        prompter: TaskPrompter
    ):
        self.gemini = gemini
        self.history = history
        self.prompter = prompter
        self.logger = logging.getLogger(f"{__name__}.GeminiFixer")

    def suggest_fix(
        self,
        code: str,
        error: Optional[Dict[str, Any]],
        stdout: str,
        stderr: str
    ) -> Optional[str]:
        """
        Возвращает исправленный код или None.
        Автоматически добавляет запрос и ответ в историю (в ветку).
        """
        if not error:
            return None
        err_msg = f"{error.get('ename')}: {error.get('evalue')}"
        traceback = "\n".join(error.get('traceback', []))
        fix_request = f"""
Предыдущий код вызвал ошибку при выполнении.

Ошибка:
{err_msg}

Traceback:
{traceback}

Stdout:
{stdout or '(пусто)'}

Stderr:
{stderr or '(пусто)'}

Код, вызвавший ошибку:
```python
{code}
```

Исправь ошибку и верни полный исправленный код.
""".strip()
        self.logger.info("🔧 Запрос исправления кода у модели...")
        full_prompt, gen_kwargs = self.prompter.compose(
            base_user_text=fix_request,
            label="r_code",
            need_conclusion=False
        )
        history_for_request = self.history.with_next_user(full_prompt)
        self.logger.debug(f"USER (fix request):\n{full_prompt[:800]}...")
        response = self.gemini.send_message(history_for_request, **gen_kwargs)
        self.logger.debug(f"MODEL (fix response):\n{response[:800]}...")
        import re
        t = response.strip()
        m = re.search(r'```[a-zA-Z0-9_-]*\s*\n(.*?)\n```', t, flags=re.S)
        if m:
            t = m.group(1)
        else:
            m2 = re.search(r'```[a-zA-Z0-9_-]*\s*(.*?)\s*```', t, flags=re.S)
            if m2:
                t = m2.group(1)
        t = re.sub(r'^\s*python\s*\r?\n', '', t, flags=re.I)
        cleaned = t.strip()
        self.history.add_user(full_prompt, meta={"kind": "fix-request"})
        self.history.add_model(cleaned, meta={"kind": "fix-response"})
        return cleaned

class NotebookSolver:
    """
    Главный класс-оркестратор для решения Jupyter Notebook с помощью Gemini.

    Процесс:
    1. Разбивает ноутбук на сегменты задач (через NotebookTaskSplitter)
    2. Для каждого сегмента:
       - info: копирует ячейки как есть, добавляет в историю диалога
       - n_code/r_code: запрашивает код, выполняет, исправляет ошибки в ветке,
         добавляет результат в историю, опционально запрашивает выводы
       - math/conclusion: запрашивает текст, добавляет markdown-ячейку
    3. Собирает итоговый ноутбук и сохраняет

    Все сообщения и ответы логируются в консоль + файл.
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        task_splitter: NotebookTaskSplitter,
        code_runner: CodeRunner,
        prompter: TaskPrompter,
        log_level: int = logging.INFO,
        log_file: Optional[str] = None,
        max_code_fix_attempts: int = 3,
        exec_timeout: float = 60.0
    ):
        """
        Args:
            gemini_client: клиент для общения с Gemini API
            task_splitter: парсер и разметчик ноутбука
            code_runner: исполнитель кода в Jupyter kernel
            prompter: генератор промптов под каждую задачу
            log_level: уровень логирования в консоль
            log_file: путь к файлу логов (если None — только консоль)
            max_code_fix_attempts: максимум попыток исправления ошибки кода
            exec_timeout: таймаут выполнения ячейки (в секундах)
        """
        self.gemini = gemini_client
        self.splitter = task_splitter
        self.runner = code_runner
        self.prompter = prompter
        self.max_code_fix_attempts = max_code_fix_attempts
        self.exec_timeout = exec_timeout

        self.history = ConversationHistory(log_level=log_level)

        self.fixer = GeminiFixer(self.gemini, self.history, self.prompter)

        self.executor = LastCellExecutor(
            runner=self.runner,
            fixer=None,
            log_level=log_level
        )

        self.logger = self._setup_logger(log_level, log_file)

        self.result_cells: List[Dict[str, Any]] = []

    def _setup_logger(self, log_level: int, log_file: Optional[str]) -> logging.Logger:
        """Настраивает логгер: только консоль, без записи в файл."""
        logger = logging.getLogger(f"{__name__}.NotebookSolver")
        return logger

    def _save_notebook_py(self, output_py_path: str):
        """
        Сохраняет текущее состояние result_cells в .py формате с маркерами ячеек.
        Markdown/raw превращаются в комментарии.
        """
        lines = [
            "# -*- coding: utf-8 -*-",
            "# Auto-exported from NotebookSolver during run. Do not edit.",
            "",
        ]
        for idx, cell in enumerate(self.result_cells):
            ctype = cell.get('cell_type', 'raw')
            lines.append(f"# ===== CELL {idx} | {ctype} =====")
            src = cell.get('source') or ""
            if ctype in ('markdown', 'raw'):
                for line in src.splitlines():
                    lines.append("# " + line)
            else:
                lines.append(src)
            lines.append("")

        text = "\n".join(lines)
        with open(output_py_path, 'w', encoding='utf-8') as f:
            f.write(text)
        self.logger.info(f"📝 Снимок .py сохранён: {output_py_path}")

    def solve(self, ipynb_path: str, output_path: Optional[str] = None) -> str:
        """
        Решает ноутбук и сохраняет результат.
        По ходу выполнения сохраняет .py-снимок текущего состояния (для отладки).
        """
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 СТАРТ РЕШЕНИЯ НОУТБУКА: {ipynb_path}")
        self.logger.info("=" * 80)

        start_time = time.time()

        if output_path is None:
            p = Path(ipynb_path)
            planned_ipynb = p.parent / f"{p.stem}_solved{p.suffix}"
        else:
            planned_ipynb = Path(output_path)
        planned_py = planned_ipynb.with_suffix(".py")

        try:
            self.logger.info("🔍 Разбиение ноутбука на сегменты задач...")
            segments = self.splitter.segment_notebook(ipynb_path)
            self.logger.info(f"✓ Получено сегментов: {len(segments)}")
            for i, seg in enumerate(segments, 1):
                self.logger.info(f"  [{i}] {seg['label']} ({len(seg['cells'])} ячеек)")

            self.result_cells.clear()

            for idx, segment in enumerate(segments, start=1):
                self.logger.info("")
                self.logger.info("=" * 80)
                self.logger.info(f"📌 СЕГМЕНТ {idx}/{len(segments)}: {segment['label']}")
                self.logger.info("=" * 80)

                self._process_segment(segment)

                self._save_notebook_py(str(planned_py))

            final_ipynb_path = str(planned_ipynb) if output_path is None else str(planned_ipynb)
            self._save_notebook(final_ipynb_path)

            self._save_notebook_py(str(planned_py))

            elapsed = time.time() - start_time
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info(f"✅ РЕШЕНИЕ ЗАВЕРШЕНО за {elapsed:.1f} сек")
            self.logger.info(f"💾 Результат: {final_ipynb_path}")
            self.logger.info("=" * 80)

            return final_ipynb_path

        except Exception:
            try:
                self._save_notebook_py(str(planned_py))
                self.logger.info(f"📝 Снимок .py сохранён перед остановкой: {planned_py}")
            except Exception as _:
                pass
            raise

    def _process_segment(self, segment: Dict[str, Any]):
        """Маршрутизирует обработку сегмента в зависимости от label."""
        label = segment['label']

        if label == 'info':
            self._process_info(segment)
        elif label == 'n_code':
            self._process_n_code(segment)
        elif label == 'r_code':
            self._process_r_code(segment)
        elif label == 'math':
            self._process_math(segment)
        elif label == 'conclusion':
            self._process_conclusion(segment)
        else:
            self.logger.warning(f"⚠ Неизвестный label: {label}. Копируем как есть.")
            for cell in segment['cells']:
                self.result_cells.append(self._copy_cell(cell))

    def _process_info(self, segment: Dict[str, Any]):
        """Info-сегмент: копируем ячейки и добавляем текст в историю для контекста."""
        self.logger.info("ℹ️ Обработка info-сегмента (контекст для модели)...")

        self.history.add_user(segment['text'], meta={"kind": "info"})
        self.logger.debug(f"USER (info):\n{self._short(segment['text'], 600)}")

        for cell in segment['cells']:
            self.result_cells.append(self._copy_cell(cell))

        self.logger.info(f"✓ Скопировано ячеек: {len(segment['cells'])}")

    def _process_n_code(self, segment: Dict[str, Any]):
        """Новый код: исполняем все код-ячейки сегмента по порядку,
        затем добавляем и исполняем новую ячейку с кодом от модели как последнюю.
        """
        self.logger.info("💻 Обработка n_code-сегмента (новый код)...")

        seg_cells = segment['cells']

        support_code_sources = [c['source'] for c in seg_cells if c['type'] == 'code']

        code_from_model = self._request_code(segment, is_new=True)

        global_preamble = self._collect_preamble_code()
        exec_cells = global_preamble + support_code_sources + [code_from_model]

        exec_result = self._execute_code_with_fixes(exec_cells)

        for cell in seg_cells:
            self.result_cells.append(self._copy_cell(cell))

        self.result_cells.append({
            'cell_type': 'code',
            'source': exec_result['final_code'],
            'metadata': {'generated': True, 'task': 'n_code'},
            'execution_count': None,
            'outputs': []
        })

        self._add_execution_to_history(exec_result['execution'])

        if segment.get('need_conclusion'):
            self.logger.info("📝 Запрос текстовых выводов по коду...")
            conclusion = self._request_conclusion_for_code(exec_result['execution'])
            self.result_cells.append({
                'cell_type': 'markdown',
                'source': conclusion,
                'metadata': {'generated': True, 'task': 'code-conclusion'}
            })

    def _process_r_code(self, segment: Dict[str, Any]):
        self.logger.info("🔧 Обработка r_code-сегмента (исправление кода)...")

        seg_cells = segment['cells']
        code_idxs = [i for i, c in enumerate(seg_cells) if c['type'] == 'code']

        if not code_idxs:
            self.logger.warning("⚠ r_code-сегмент без code-ячейки — копируем как есть.")
            for cell in seg_cells:
                self.result_cells.append(self._copy_cell(cell))
            return

        target_abs = segment.get('rcode_target', None)
        target_local_idx = None
        if isinstance(target_abs, int):
            for i, c in enumerate(seg_cells):
                if c.get('index') == target_abs and c.get('type') == 'code':
                    target_local_idx = i
                    break
        if target_local_idx is None:
            target_local_idx = code_idxs[-1]

        support_code_sources = [seg_cells[i]['source'] for i in code_idxs if i < target_local_idx]

        code_segment = {'text': seg_cells[target_local_idx]['source']}
        code_from_model = self._request_code(code_segment, is_new=False)

        global_preamble = self._collect_preamble_code()
        exec_cells = global_preamble + support_code_sources + [code_from_model]

        exec_result = self._execute_code_with_fixes(exec_cells)

        for i, cell in enumerate(seg_cells):
            if i == target_local_idx:
                self.result_cells.append({
                    'cell_type': 'code',
                    'source': exec_result['final_code'],
                    'metadata': {'generated': True, 'revised': True, 'task': 'r_code'},
                    'execution_count': None,
                    'outputs': []
                })
            else:
                self.result_cells.append(self._copy_cell(cell))

        self._add_execution_to_history(exec_result['execution'])

        if segment.get('need_conclusion'):
            self.logger.info("📝 Запрос текстовых выводов по коду...")
            conclusion = self._request_conclusion_for_code(exec_result['execution'])
            self.result_cells.append({
                'cell_type': 'markdown',
                'source': conclusion,
                'metadata': {'generated': True, 'task': 'code-conclusion'}
            })

    def _process_math(self, segment: Dict[str, Any]):
        """Математическое решение: запрашиваем, добавляем markdown-ячейку."""
        self.logger.info("🧮 Обработка math-сегмента...")

        for cell in segment['cells']:
            self.result_cells.append(self._copy_cell(cell))

        prompt, gen_kwargs = self.prompter.compose(
            base_user_text=segment['text'],
            label='math'
        )

        history = self.history.with_next_user(prompt)
        self.logger.debug(f"USER (math):\n{self._short(prompt, 600)}")

        response = self.gemini.send_message(history, **gen_kwargs)
        self.logger.debug(f"MODEL (math):\n{self._short(response, 600)}")

        self.history.add_user(prompt, meta={"kind": "math-request"})
        self.history.add_model(response, meta={"kind": "math"})

        import re
        response_for_nb = re.sub(r'(?<!\n)\n(?!\n)', '\n\n', response)

        self.result_cells.append({
            'cell_type': 'markdown',
            'source': response_for_nb,
            'metadata': {'generated': True, 'task': 'math'}
        })

    def _process_conclusion(self, segment: Dict[str, Any]):
        """Выводы: запрашиваем, добавляем markdown-ячейку."""
        self.logger.info("📊 Обработка conclusion-сегмента...")

        for cell in segment['cells']:
            self.result_cells.append(self._copy_cell(cell))

        prompt, gen_kwargs = self.prompter.compose(
            base_user_text=segment['text'],
            label='conclusion'
        )

        history = self.history.with_next_user(prompt)
        self.logger.debug(f"USER (conclusion):\n{self._short(prompt, 600)}")

        response = self.gemini.send_message(history, **gen_kwargs)
        self.logger.debug(f"MODEL (conclusion):\n{self._short(response, 600)}")

        self.history.add_user(prompt, meta={"kind": "conclusion-request"})
        self.history.add_model(response, meta={"kind": "conclusion"})

        self.result_cells.append({
            'cell_type': 'markdown',
            'source': response,
            'metadata': {'generated': True, 'task': 'conclusion'}
        })

    def _request_code(self, segment: Dict[str, Any], is_new: bool) -> str:
        label = 'n_code' if is_new else 'r_code'
        prompt, gen_kwargs = self.prompter.compose(
            base_user_text=segment['text'],
            label=label,
            need_conclusion=False
        )
        history = self.history.with_next_user(prompt)
        self.logger.debug(f"USER ({label}):\n{self._short(prompt, 800)}")
        response = self.gemini.send_message(history, **gen_kwargs)
        self.logger.debug(f"MODEL ({label}):\n{self._short(response, 800)}")
        import re
        t = response.strip()
        m = re.search(r'```[a-zA-Z0-9_-]*\s*\n(.*?)\n```', t, flags=re.S)
        if m:
            t = m.group(1)
        else:
            m2 = re.search(r'```[a-zA-Z0-9_-]*\s*(.*?)\s*```', t, flags=re.S)
            if m2:
                t = m2.group(1)
        t = re.sub(r'^\s*python\s*\r?\n', '', t, flags=re.I)
        code = t.strip()
        self.history.add_user(prompt, meta={"kind": f"{label}-request"})
        self.history.add_model(code, meta={"kind": "code", "label": label})
        return code

    def _execute_code_with_fixes(self, cells: List[str]) -> Dict[str, Any]:
        """
        Выполняет код с автоисправлением ошибок.
        ВАЖНО: пролог (все предыдущие code-ячейки) исполняется ИНКРЕМЕНТАЛЬНО и ровно один раз за сессию ядра.
            При попытках фикса переисполняется только последняя ячейка.
        """
        if not cells:
            raise ValueError("Нужна хотя бы одна ячейка для выполнения.")

        self.logger.info("▶️ Выполнение кода с возможностью автоисправления...")

        self.history.open_branch(anchor=None, anchor_role='model')

        last_code = cells[-1]
        attempt = 0
        result = None

        try:
            while attempt <= self.max_code_fix_attempts:
                attempt += 1
                self.logger.info(f"⚙️ Попытка #{attempt}/{self.max_code_fix_attempts + 1}...")

                all_cells = cells[:-1] + [last_code]

                result = self.executor.run(
                    cells=all_cells,
                    prepare_strategy='auto', 
                    max_fixes=0
                )

                if result.success:
                    self.logger.info(f"✅ Код выполнен успешно (попытка #{attempt})")
                    self.history.commit_branch(squash=True, keep_last_exec_summary=False)
                    return {
                        "success": True,
                        "final_code": result.final_code,
                        "execution": result.execution,
                        "attempts": attempt
                    }

                if attempt > self.max_code_fix_attempts:
                    self.logger.error(f"❌ Не удалось исправить код за {attempt} попыток")
                    break

                self.logger.warning(f"⚠️ Ошибка выполнения: {result.execution.error.get('ename')}")
                self.logger.info("🔧 Запрос исправления...")

                fixed_code = self.fixer.suggest_fix(
                    code=last_code,
                    error=result.execution.error,
                    stdout=result.execution.stdout,
                    stderr=result.execution.stderr
                )

                if not fixed_code or fixed_code.strip() == last_code.strip():
                    self.logger.warning("⚠️ Фиксер не предложил изменений. Останавливаемся.")
                    break

                last_code = fixed_code

            self.logger.error("❌ Не удалось получить рабочий код")
            self.history.commit_branch(squash=True, keep_last_exec_summary=False)
            return {
                "success": False,
                "final_code": last_code,
                "execution": result.execution if result else None,
                "attempts": attempt
            }

        except Exception as e:
            self.logger.error(f"💥 Критическая ошибка при выполнении: {e}")
            self.history.discard_branch()
            raise

    def _add_execution_to_history(self, exec_res: ExecutionResult):
        """
        Добавляет сводку выполнения в историю как user-сообщение и прикладывает изображения
        (image/png, image/jpeg) в виде inline_data, чтобы модель реально их «видела».
        """
        import base64

        parts_text = []

        parts_text.append("[Результат выполнения предыдущей ячейки]")
        parts_text.append(f"Время: {exec_res.elapsed_sec:.2f} сек")

        if exec_res.stdout:
            parts_text.append("\nВывод (stdout):")
            parts_text.append(self._truncate(exec_res.stdout, 2000))

        if exec_res.stderr:
            parts_text.append("\nСтандартная ошибка (stderr):")
            parts_text.append(self._truncate(exec_res.stderr, 1000))

        if exec_res.result_text:
            parts_text.append("\nРезультат последнего выражения:")
            parts_text.append(self._truncate(exec_res.result_text, 1000))

        total_imgs = len(exec_res.images)
        if total_imgs:
            parts_text.append(f"\nСоздано изображений: {total_imgs}")

        text_blob = "\n".join(parts_text)

        parts: List[Any] = [text_blob]

        allowed_mimes = {"image/png", "image/jpeg"}
        max_images = 3
        attached = 0

        for img in exec_res.images:
            if attached >= max_images:
                break
            if img.mime_type not in allowed_mimes:
                continue
            try:
                b64 = base64.b64encode(img.data).decode("ascii")
                parts.append({
                    "inline_data": {
                        "mime_type": img.mime_type,
                        "data": b64
                    }
                })
                attached += 1
            except Exception as e:
                self.logger.debug(f"Не удалось прикрепить изображение ({img.mime_type}): {e}")

        meta = {"kind": "exec-result", "images_total": total_imgs, "images_attached": attached}
        self.history.add_user_parts(parts, meta=meta)
        self.logger.debug(f"Добавлен результат выполнения в историю: {len(text_blob)} симв., изображений приложено: {attached}")

    def _request_conclusion_for_code(self, exec_res: ExecutionResult) -> str:
        """
        Запрашивает выводы по результатам предыдущей ячейки.
        Текст запроса и стиль полностью формируются в TaskPrompter.
        """
        prompt_full, gen_kwargs = self.prompter.compose(
            base_user_text="",
            label="code_conclusion"
        )

        history = self.history.with_next_user(prompt_full)
        self.logger.debug(f"USER (conclusion for code):\n{self._short(prompt_full, 400)}")

        response = self.gemini.send_message(history, **gen_kwargs)
        self.logger.debug(f"MODEL (conclusion for code):\n{self._short(response, 400)}")

        self.history.add_user(prompt_full, meta={"kind": "code-conclusion-request"})
        self.history.add_model(response, meta={"kind": "code-conclusion"})

        return response

    def _collect_preamble_code(self) -> List[str]:
        """Собирает все code-ячейки из уже обработанных сегментов (для пролога)."""
        preamble = []
        for cell in self.result_cells:
            if cell.get('cell_type') == 'code':
                preamble.append(cell['source'])
        return preamble

    def _copy_cell(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        """Копирует ячейку из формата splitter'а в формат nbformat."""
        ctype = cell['type']
        if ctype == 'code':
            return {
                'cell_type': 'code',
                'source': cell['source'],
                'metadata': {},
                'execution_count': None,
                'outputs': []
            }
        elif ctype == 'markdown':
            return {
                'cell_type': 'markdown',
                'source': cell['source'],
                'metadata': {}
            }
        else:
            return {
                'cell_type': 'raw',
                'source': cell['source'],
                'metadata': {}
            }

    def _save_notebook(self, output_path: str):
        """Сохраняет итоговый ноутбук в .ipynb формате."""
        self.logger.info(f"💾 Сохранение результата в: {output_path}")

        nb = nbformat.v4.new_notebook()

        for cell_dict in self.result_cells:
            if cell_dict['cell_type'] == 'code':
                cell = nbformat.v4.new_code_cell(
                    source=cell_dict['source'],
                    metadata=cell_dict.get('metadata', {})
                )
            elif cell_dict['cell_type'] == 'markdown':
                cell = nbformat.v4.new_markdown_cell(
                    source=cell_dict['source'],
                    metadata=cell_dict.get('metadata', {})
                )
            else:
                cell = nbformat.v4.new_raw_cell(
                    source=cell_dict['source'],
                    metadata=cell_dict.get('metadata', {})
                )
            nb.cells.append(cell)

        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        self.logger.info(f"✓ Ноутбук сохранён ({len(nb.cells)} ячеек)")

    @staticmethod
    def _short(s: str, limit: int = 300) -> str:
        """Обрезает строку для превью в логах."""
        s = (s or "").replace("\n", "\\n")
        return s if len(s) <= limit else (s[:limit - 3] + "...")

    @staticmethod
    def _truncate(s: str, limit: int) -> str:
        """Обрезает строку с многоточием."""
        s = s or ""
        return s if len(s) <= limit else (s[:limit - 3] + "...")

