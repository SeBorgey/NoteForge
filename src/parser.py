import json
import logging
from typing import Any, Dict, List, Tuple

import nbformat

from client import GeminiClient


class NotebookTaskSplitter:
    """
    Класс для парсинга Jupyter Notebook (.ipynb), конвертации в "плоский" .py
    и разбиения на сегменты задач с помощью модели Gemini.

    Возвращает удобную структуру:
    [
      {
        "label": "info" | "n_code" | "r_code" | "math" | "conclusion",
        "text": "<конкатенация исходного текста ячеек>",
        "cells": [
          {"index": int, "type": "markdown"|"code"|"raw", "source": str},
          ...
        ],
        "need_conclusion": bool  # только для n_code/r_code; для остальных False
      },
      ...
    ]
    """

    ALLOWED_LABELS = ("info", "n_code", "r_code", "math", "conclusion")

    def __init__(
        self,
        gemini_client: "GeminiClient",
        log_level: int = logging.INFO,
        temperature: float = 0.2,
        top_p: float = 0.1,
        max_output_tokens: int = 8192,
    ):
        self.gemini = gemini_client
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens

        self.logger = logging.getLogger(f"{__name__}.NotebookTaskSplitter")

    def segment_notebook(self, ipynb_path: str) -> List[Dict[str, Any]]:
        """
        Разбивает ноутбук на сегменты задач и возвращает структуру,
        удобную для последующей работы в Python.

        Args:
            ipynb_path: путь к .ipynb файлу

        Returns:
            Список сегментов со структурой, описанной в классе.
        """
        self.logger.info(f"➤ Старт разметки ноутбука: {ipynb_path}")

        nb = self._read_notebook(ipynb_path)
        cells = self._extract_cells(nb)
        self.logger.info(
            f"✓ Зачитано ячеек: {len(cells)} "
            f"(code={sum(c['type'] == 'code' for c in cells)}, "
            f"markdown={sum(c['type'] == 'markdown' for c in cells)}, "
            f"raw={sum(c['type'] == 'raw' for c in cells)})"
        )

        py_text = self._to_py_with_markers(cells)
        self.logger.debug(
            "Превью py-текста с маркерами:\n"
            + py_text[:1000]
            + ("..." if len(py_text) > 1000 else "")
        )

        prompt = self._build_prompt(py_text, len(cells))

        response_text = self.gemini.send_message(
            prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            response_mime_type="application/json",
        )

        self.logger.debug(f"Сырой ответ модели (превью): {response_text[:500]}...")
        model_data = self._parse_json(response_text)

        segments = self._normalize_and_cover_segments(
            model_data=model_data, total_cells=len(cells)
        )

        result = self._build_result_segments(segments, cells)
        self.logger.info(f"✓ Получено сегментов: {len(result)}")
        return result

    def _read_notebook(self, path: str):
        try:
            nb = nbformat.read(path, as_version=4)
            return nb
        except Exception as e:
            self.logger.error(f"✗ Ошибка чтения .ipynb: {e}")
            raise

    def _extract_cells(self, nb) -> List[Dict[str, Any]]:
        cells = []
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type not in ("code", "markdown", "raw"):
                ctype = "raw"
            else:
                ctype = cell.cell_type

            source = cell.get("source", "")
            cells.append(
                {
                    "index": idx,
                    "type": ctype,
                    "source": source if isinstance(source, str) else "\n".join(source),
                }
            )
        return cells

    def _to_py_with_markers(self, cells: List[Dict[str, Any]]) -> str:
        """
        Делаем .py-представление: кодовые ячейки - как есть,
        markdown/raw - превращаем в комментарии,
        и обязательно ставим маркеры ячеек.
        """
        lines = [
            "# -*- coding: utf-8 -*-",
            "# Auto-generated from .ipynb for segmentation. Do not edit.",
            "",
        ]
        for c in cells:
            header = f"# ===== CELL {c['index']} | {c['type']} ====="
            lines.append(header)
            src = c["source"] or ""

            if c["type"] in ("markdown", "raw"):
                for line in src.splitlines():
                    lines.append("# " + line)
            else:  # code
                lines.append(src)

            lines.append("")

        return "\n".join(lines)

    def _build_prompt(self, py_text: str, total_cells: int) -> str:
        """
        Просим модель вернуть только JSON с последовательными сегментами.
        """
        labels_desc = (
            "- info: нет задания, просто информационные ячейки\n"
            "- n_code: нужно написать новый код, добавив новую ячейку (или используя плейсхолдер-код, если он есть)\n"
            "- r_code: нужно дописать/исправить код в существующей ячейке\n"
            "- math: математическое решение/доказательство\n"
            "- conclusion: ответить на вопрос / сделать выводы"
        )

        example = {
            "segments": [
                {"label": "info", "cell_indices": [0]},
                {"label": "r_code", "cell_indices": [1, 2], "need_conclusion": True, "rcode_target": 2},
                {"label": "conclusion", "cell_indices": [3]}
            ]
        }

        instr = f"""
Ты получаешь текстовый .py, сгенерированный из Jupyter Notebook. Каждая ячейка помечена маркером:
# ===== CELL <номер> | <тип> =====
где <тип> ∈ {{code, markdown, raw}}.

Задача: разбить последовательно весь ноутбук на сегменты пяти типов:
{labels_desc}
В одном сегменте может быть несколько ячеек.

Пример:
Md: Напиши код
Code: Какой-то готовый к использованию код
Code: Код с плейсхолдером # TODO

Такой случай объединяй в один сегмент r_code.

Требования:
- Сегменты должны покрывать все {total_cells} ячеек без пропусков.
- Сегменты не должны перекрываться.
- Сегменты идут в порядке возрастания индексов.
- Каждый сегмент — непрерывный диапазон по индексам.
- Не создавай пустые сегменты.

Дополнительно для 'r_code':
- Добавь целочисленное поле "rcode_target" — абсолютный индекс ячейки ноутбука, которую нужно дописать.
- Значение rcode_target обязательно должно входить в массив cell_indices этого сегмента и указывать на ячейку типа code.
- Если в сегменте несколько ячеек требуют правки, выбери последнюю из них.

Для 'n_code' и 'r_code' добавь булев флаг "need_conclusion": true/false — требуется ли после кода сделать выводы, если отдельной ячейки для выводов нет.

Формат ответа: верни строго JSON с ключом "segments", без комментариев, без markdown и без текста ячеек:
{json.dumps(example, ensure_ascii=False, indent=2)}

Где:
- label ∈ ["info","n_code","r_code","math","conclusion"]
- cell_indices — массив индексов ячеек (целые числа) в непрерывном диапазоне
- need_conclusion — обязательно только для 'n_code'/'r_code'
- rcode_target — обязателен только для 'r_code' и должен быть одним из cell_indices

Ниже идет содержимое файла (.py) с маркерами ячеек. Выполни разметку, учитывая границы ячеек и их содержание.

----- BEGIN PY TEXT -----
{py_text}
----- END PY TEXT -----
""".strip()

        return instr

    def _parse_json(self, response_text: str) -> Dict[str, Any]:
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            self.logger.error(f"✗ Ответ модели не JSON: {e}")
            raise

        if isinstance(data, list):
            data = {"segments": data}

        if not isinstance(data, dict) or "segments" not in data:
            raise ValueError("Ответ модели должен быть объектом с ключом 'segments'.")

        if not isinstance(data["segments"], list):
            raise ValueError("'segments' должен быть массивом.")

        return data

    def _normalize_and_cover_segments(self, model_data: Dict[str, Any], total_cells: int) -> List[Dict[str, Any]]:
        raw_segments = model_data["segments"]
        normalized = []

        for i, seg in enumerate(raw_segments, start=1):
            if not isinstance(seg, dict):
                raise ValueError(f"Элемент segments[{i}] должен быть объектом.")
            label = seg.get("label")
            indices = seg.get("cell_indices")
            if label not in self.ALLOWED_LABELS:
                raise ValueError(f"Недопустимый label '{label}' в segments[{i}].")
            if not isinstance(indices, list) or not indices:
                raise ValueError(f"'cell_indices' должен быть непустым списком в segments[{i}].")

            cleaned = []
            for x in indices:
                if not isinstance(x, int):
                    raise ValueError(f"Индекс {x} не int в segments[{i}].")
                if not (0 <= x < total_cells):
                    raise ValueError(f"Индекс {x} вне диапазона [0..{total_cells - 1}] в segments[{i}].")
                cleaned.append(x)

            need_conclusion = False
            if label in ("n_code", "r_code"):
                nc = seg.get("need_conclusion", False)
                if isinstance(nc, bool):
                    need_conclusion = nc
                else:
                    if nc in (0, 1):
                        need_conclusion = bool(nc)
                        self.logger.warning(f"⚠ need_conclusion в segments[{i}] приведён к bool из {nc}.")
                    elif isinstance(nc, str) and nc.lower() in ("true", "false"):
                        need_conclusion = nc.lower() == "true"
                        self.logger.warning(f"⚠ need_conclusion в segments[{i}] приведён к bool из строки '{nc}'.")
                    elif nc is None:
                        need_conclusion = False
                    else:
                        raise ValueError(f"need_conclusion должен быть bool для n_code/r_code (segments[{i}]).")

            rcode_target = None
            if label == "r_code":
                rt = seg.get("rcode_target", None)
                if rt is not None:
                    if isinstance(rt, int) and (0 <= rt < total_cells):
                        if rt in cleaned:
                            rcode_target = rt
                        else:
                            self.logger.warning(f"⚠ rcode_target={rt} не входит в cell_indices в segments[{i}]. Поле будет опущено.")
                    else:
                        self.logger.warning(f"⚠ rcode_target некорректен в segments[{i}]. Поле будет опущено.")

            cleaned = sorted(set(cleaned))
            for rng in self._split_into_contiguous_runs(cleaned):
                rt_in_run = rcode_target if (rcode_target is not None and rcode_target in rng) else None
                normalized.append((label, rng, need_conclusion, rt_in_run))

        normalized.sort(key=lambda x: x[1][0])

        occupied = set()
        final_segments = []
        for label, idxs, need_conclusion, rt in normalized:
            if any(i in occupied for i in idxs):
                overlap = [i for i in idxs if i in occupied]
                raise ValueError(f"Перекрывающиеся сегменты на индексах: {overlap}")
            for i in idxs:
                occupied.add(i)
            final_segments.append((label, idxs, need_conclusion, rt))

        missing = [i for i in range(total_cells) if i not in occupied]
        if missing:
            self.logger.warning(f"⚠ Обнаружены пропущенные ячейки: {missing}. Будут добавлены сегменты 'info'.")
            info_runs = self._split_into_contiguous_runs(missing)
            for rng in info_runs:
                final_segments.append(("info", rng, False, None))
            final_segments.sort(key=lambda x: x[1][0])

        result = []
        for label, idxs, need_conclusion, rt in final_segments:
            item = {
                "label": label,
                "cell_indices": idxs,
                "need_conclusion": need_conclusion if label in ("n_code", "r_code") else False,
            }
            if label == "r_code" and rt is not None:
                item["rcode_target"] = rt
            result.append(item)

        return result

    def _split_into_contiguous_runs(self, sorted_unique: List[int]) -> List[List[int]]:
        if not sorted_unique:
            return []
        runs = []
        start = sorted_unique[0]
        prev = start
        for x in sorted_unique[1:]:
            if x == prev + 1:
                prev = x
            else:
                runs.append(list(range(start, prev + 1)))
                start = x
                prev = x
        runs.append(list(range(start, prev + 1)))
        return runs

    def _build_result_segments(self, segments: List[Dict[str, Any]], cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for seg in segments:
            idxs = seg["cell_indices"]
            seg_cells = [cells[i] for i in idxs]
            text = "\n\n".join(c["source"] for c in seg_cells)
            item = {
                "label": seg["label"],
                "text": text,
                "cells": seg_cells,
                "need_conclusion": bool(seg.get("need_conclusion", False)) if seg["label"] in ("n_code", "r_code") else False,
            }
            if seg["label"] == "r_code" and "rcode_target" in seg:
                item["rcode_target"] = seg["rcode_target"]
            result.append(item)
        return result
