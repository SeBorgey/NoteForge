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
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

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

        # Формируем промпт с суффиксом для r_code
        full_prompt, gen_kwargs = self.prompter.compose(
            base_user_text=fix_request,
            label="r_code",
            need_conclusion=False
        )

        # Получаем историю + новый user-запрос (не добавляя его в историю пока)
        history_for_request = self.history.with_next_user(full_prompt)

        # Логируем
        self.logger.debug(f"USER (fix request):\n{full_prompt[:800]}...")

        # Отправляем в модель
        response = self.gemini.send_message(history_for_request, **gen_kwargs)

        self.logger.debug(f"MODEL (fix response):\n{response[:800]}...")

        # Теперь добавляем ОБА сообщения в историю (в открытую ветку)
        self.history.add_user(full_prompt, meta={"kind": "fix-request"})
        self.history.add_model(response, meta={"kind": "fix-response"})

        return response


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

        # История разговора с моделью
        self.history = ConversationHistory(log_level=log_level)

        # Фиксер ошибок через Gemini
        self.fixer = GeminiFixer(self.gemini, self.history, self.prompter)

        # Executor для запуска последней ячейки с подготовкой состояния
        self.executor = LastCellExecutor(
            runner=self.runner,
            fixer=None,  # фиксер не используем внутри executor'а
            log_level=log_level
        )

        # Настройка логирования
        self.logger = self._setup_logger(log_level, log_file)

        # Накопитель результирующих ячеек
        self.result_cells: List[Dict[str, Any]] = []

    def _setup_logger(self, log_level: int, log_file: Optional[str]) -> logging.Logger:
        """Настраивает логгер с выводом в консоль и (опционально) файл."""
        logger = logging.getLogger(f"{__name__}.NotebookSolver")
        logger.setLevel(logging.DEBUG)  # внутри фильтруем handlers'ами
        logger.handlers.clear()

        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Консольный handler
        console = logging.StreamHandler()
        console.setLevel(log_level)
        console.setFormatter(formatter)
        logger.addHandler(console)

        # Файловый handler (если задан)
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # в файл пишем ВСЁ
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"📄 Полные логи сохраняются в: {log_file}")

        return logger

    # =========================================================================
    # ПУБЛИЧНЫЙ API
    # =========================================================================

    def solve(self, ipynb_path: str, output_path: Optional[str] = None) -> str:
        """
        Решает ноутбук и сохраняет результат.

        Args:
            ipynb_path: путь к входному .ipynb
            output_path: путь к выходному .ipynb (если None — добавляется суффикс _solved)

        Returns:
            путь к созданному файлу
        """
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 СТАРТ РЕШЕНИЯ НОУТБУКА: {ipynb_path}")
        self.logger.info("=" * 80)

        start_time = time.time()

        # 1. Разбиваем ноутбук на сегменты
        self.logger.info("🔍 Разбиение ноутбука на сегменты задач...")
        segments = self.splitter.segment_notebook(ipynb_path)
        self.logger.info(f"✓ Получено сегментов: {len(segments)}")
        for i, seg in enumerate(segments, 1):
            self.logger.info(f"  [{i}] {seg['label']} ({len(seg['cells'])} ячеек)")

        # 2. Обрабатываем каждый сегмент
        self.result_cells.clear()

        for idx, segment in enumerate(segments, start=1):
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info(f"📌 СЕГМЕНТ {idx}/{len(segments)}: {segment['label']}")
            self.logger.info("=" * 80)

            self._process_segment(segment)

        # 3. Сохраняем итоговый ноутбук
        if output_path is None:
            p = Path(ipynb_path)
            output_path = str(p.parent / f"{p.stem}_solved{p.suffix}")

        self._save_notebook(output_path)

        elapsed = time.time() - start_time
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"✅ РЕШЕНИЕ ЗАВЕРШЕНО за {elapsed:.1f} сек")
        self.logger.info(f"💾 Результат: {output_path}")
        self.logger.info("=" * 80)

        return output_path

    # =========================================================================
    # ОБРАБОТКА СЕГМЕНТОВ
    # =========================================================================

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

        # Добавляем текст в историю как user-сообщение
        self.history.add_user(segment['text'], meta={"kind": "info"})
        self.logger.debug(f"USER (info):\n{self._short(segment['text'], 600)}")

        # Копируем ячейки в результат
        for cell in segment['cells']:
            self.result_cells.append(self._copy_cell(cell))

        self.logger.info(f"✓ Скопировано ячеек: {len(segment['cells'])}")

    def _process_n_code(self, segment: Dict[str, Any]):
        """Написание нового кода: запрашиваем, выполняем, добавляем ячейку."""
        self.logger.info("💻 Обработка n_code-сегмента (новый код)...")

        # Копируем исходные ячейки
        for cell in segment['cells']:
            self.result_cells.append(self._copy_cell(cell))

        # Запрашиваем код у модели
        code = self._request_code(segment, is_new=True)

        # Выполняем с автоисправлением
        exec_result = self._execute_code_with_fixes([code])

        # Добавляем ячейку с кодом в результат
        self.result_cells.append({
            'cell_type': 'code',
            'source': exec_result['final_code'],
            'metadata': {'generated': True, 'task': 'n_code'},
            'execution_count': None,
            'outputs': []
        })

        # Добавляем результат выполнения в историю (для контекста следующих сегментов)
        self._add_execution_to_history(exec_result['execution'])

        # Если требуются выводы — запрашиваем их отдельно
        if segment.get('need_conclusion'):
            self.logger.info("📝 Запрос текстовых выводов по коду...")
            conclusion = self._request_conclusion_for_code(exec_result['execution'])
            self.result_cells.append({
                'cell_type': 'markdown',
                'source': conclusion,
                'metadata': {'generated': True, 'task': 'code-conclusion'}
            })

    def _process_r_code(self, segment: Dict[str, Any]):
        """Исправление/дополнение кода: запрашиваем, выполняем, заменяем ячейку."""
        self.logger.info("🔧 Обработка r_code-сегмента (исправление кода)...")

        # Копируем все ячейки, запоминая индекс последней code-ячейки
        last_code_idx = None
        for cell in segment['cells']:
            if cell['type'] == 'code':
                last_code_idx = len(self.result_cells)
            self.result_cells.append(self._copy_cell(cell))

        # Запрашиваем исправленный код
        code = self._request_code(segment, is_new=False)

        # Выполняем с автоисправлением (пролог = весь код до этого момента)
        preamble = self._collect_preamble_code()
        exec_result = self._execute_code_with_fixes(preamble + [code])

        # Заменяем последнюю code-ячейку на исправленную
        if last_code_idx is not None:
            self.result_cells[last_code_idx] = {
                'cell_type': 'code',
                'source': exec_result['final_code'],
                'metadata': {'generated': True, 'revised': True, 'task': 'r_code'},
                'execution_count': None,
                'outputs': []
            }
        else:
            # Если code-ячейки не было — добавляем новую
            self.logger.warning("⚠ Не найдена code-ячейка для замены. Добавляем новую.")
            self.result_cells.append({
                'cell_type': 'code',
                'source': exec_result['final_code'],
                'metadata': {'generated': True, 'task': 'r_code'},
                'execution_count': None,
                'outputs': []
            })

        # Добавляем результат в историю
        self._add_execution_to_history(exec_result['execution'])

        # Если нужны выводы
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

        # Копируем исходные ячейки
        for cell in segment['cells']:
            self.result_cells.append(self._copy_cell(cell))

        # Формируем промпт
        prompt, gen_kwargs = self.prompter.compose(
            base_user_text=segment['text'],
            label='math'
        )

        # Отправляем в модель
        history = self.history.with_next_user(prompt)
        self.logger.debug(f"USER (math):\n{self._short(prompt, 600)}")

        response = self.gemini.send_message(history, **gen_kwargs)
        self.logger.debug(f"MODEL (math):\n{self._short(response, 600)}")

        # Добавляем в историю
        self.history.add_user(prompt, meta={"kind": "math-request"})
        self.history.add_model(response, meta={"kind": "math"})

        # Добавляем markdown-ячейку с решением
        self.result_cells.append({
            'cell_type': 'markdown',
            'source': response,
            'metadata': {'generated': True, 'task': 'math'}
        })

    def _process_conclusion(self, segment: Dict[str, Any]):
        """Выводы: запрашиваем, добавляем markdown-ячейку."""
        self.logger.info("📊 Обработка conclusion-сегмента...")

        # Копируем исходные ячейки
        for cell in segment['cells']:
            self.result_cells.append(self._copy_cell(cell))

        # Формируем промпт
        prompt, gen_kwargs = self.prompter.compose(
            base_user_text=segment['text'],
            label='conclusion'
        )

        # Отправляем в модель
        history = self.history.with_next_user(prompt)
        self.logger.debug(f"USER (conclusion):\n{self._short(prompt, 600)}")

        response = self.gemini.send_message(history, **gen_kwargs)
        self.logger.debug(f"MODEL (conclusion):\n{self._short(response, 600)}")

        # Добавляем в историю
        self.history.add_user(prompt, meta={"kind": "conclusion-request"})
        self.history.add_model(response, meta={"kind": "conclusion"})

        # Добавляем markdown-ячейку
        self.result_cells.append({
            'cell_type': 'markdown',
            'source': response,
            'metadata': {'generated': True, 'task': 'conclusion'}
        })

    # =========================================================================
    # РАБОТА С КОДОМ
    # =========================================================================

    def _request_code(self, segment: Dict[str, Any], is_new: bool) -> str:
        """Запрашивает код у модели (новый или исправленный)."""
        label = 'n_code' if is_new else 'r_code'

        prompt, gen_kwargs = self.prompter.compose(
            base_user_text=segment['text'],
            label=label,
            need_conclusion=False  # выводы запросим после выполнения
        )

        history = self.history.with_next_user(prompt)
        self.logger.debug(f"USER ({label}):\n{self._short(prompt, 800)}")

        response = self.gemini.send_message(history, **gen_kwargs)
        self.logger.debug(f"MODEL ({label}):\n{self._short(response, 800)}")

        # Добавляем в историю
        self.history.add_user(prompt, meta={"kind": f"{label}-request"})
        self.history.add_model(response, meta={"kind": "code", "label": label})

        return response

    def _execute_code_with_fixes(self, cells: List[str]) -> Dict[str, Any]:
        """
        Выполняет код с автоисправлением ошибок через ветки истории.

        Теперь НЕ перезапускает ядро и НЕ переисполняет пролог: prepare_strategy='never'.
        Исполняется только последняя ячейка в текущем состоянии ядра.
        Таймаут исполнения (если сработает) пробрасывается как исключение и НЕ отправляется в LLM.
        """
        if not cells:
            raise ValueError("Нужна хотя бы одна ячейка для выполнения.")

        self.logger.info("▶️ Выполнение кода с возможностью автоисправления...")

        # Открываем ветку (якорь — последнее сообщение модели, т.е. код)
        self.history.open_branch(anchor=None, anchor_role='model')

        last_code = cells[-1]
        attempt = 0
        result = None

        try:
            while attempt <= self.max_code_fix_attempts:
                attempt += 1
                self.logger.info(f"⚙️ Попытка #{attempt}/{self.max_code_fix_attempts + 1}...")

                # Собираем список ячеек: пролог + последняя (пролог НЕ будет выполнен при prepare_strategy='never')
                all_cells = cells[:-1] + [last_code]

                # Критично: НЕ перезапускаем ядро и НЕ исполняем пролог
                result = self.executor.run(
                    cells=all_cells,
                    prepare_strategy='never',  # было 'auto'
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

                # Ошибка — пытаемся исправить (кроме таймаута: он сюда не попадёт, т.к. пробрасывается исключением)
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

            # Если сюда дошли — не удалось исправить
            self.logger.error("❌ Не удалось получить рабочий код")
            self.history.commit_branch(squash=True, keep_last_exec_summary=False)
            return {
                "success": False,
                "final_code": last_code,
                "execution": result.execution if result else None,
                "attempts": attempt
            }

        except Exception as e:
            # Включая TimeoutError — не отправляем это в LLM, просто останавливаемся
            self.logger.error(f"💥 Критическая ошибка при выполнении: {e}")
            self.history.discard_branch()
            raise

    def _add_execution_to_history(self, exec_res: ExecutionResult):
        """
        Добавляет сводку выполнения в историю как user-сообщение и прикладывает изображения
        (image/png, image/jpeg) в виде inline_data, чтобы модель реально их «видела».
        """
        # Локальный импорт, чтобы не трогать верхние импорты
        import base64

        # 1) Текстовая сводка
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

        # 2) Собираем parts: сначала текст, далее картинки как inline_data
        parts: List[Any] = [text_blob]

        # Правила вложения картинок
        allowed_mimes = {"image/png", "image/jpeg"}
        max_images = 3  # можно настроить
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
        Запрашивает выводы только по фактическим результатам выполнения предыдущей ячейки.
        Модель увидит сводку выполнения (stdout/stderr/result_text/счётчик изображений) из истории,
        поэтому здесь просим краткие, приземлённые выводы без домыслов.
        """
        prompt = (
            "Сформулируй краткие выводы по результатам выполнения предыдущей ячейки. "
            "Опираться только на фактический вывод (stdout, stderr, результат последнего выражения) "
            "и на явные подсказки/описания графиков, если они были. "
            "Если данных недостаточно — скажи об этом."
        )

        prompt_full, gen_kwargs = self.prompter.compose(
            base_user_text=prompt,
            label='code_conclusion'
        )

        history = self.history.with_next_user(prompt_full)
        self.logger.debug(f"USER (conclusion for code):\n{self._short(prompt_full, 400)}")

        response = self.gemini.send_message(history, **gen_kwargs)
        self.logger.debug(f"MODEL (conclusion for code):\n{self._short(response, 400)}")

        # Добавляем в историю
        self.history.add_user(prompt_full, meta={"kind": "code-conclusion-request"})
        self.history.add_model(response, meta={"kind": "code-conclusion"})

        return response

    # =========================================================================
    # ВСПОМОГАТЕЛЬНОЕ
    # =========================================================================

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
        else:  # raw
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
            else:  # raw
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

