import logging
from typing import Any, Dict, Optional, Tuple


class TaskPrompter:
    """
    Формирует «приписку» (tail/suffix) к пользовательскому сообщению в зависимости от типа задачи.
    Это НЕ системный промпт — просто добавка к концу промпта.

      - Для code-задач (n_code/r_code) — только код, без Markdown/объяснений/«выводов».
      - Для math — формульное решение.
      - Для conclusion — краткий финальный ответ.
      - Для info — коротко, по делу.
      - Для code_conclusion — отдельные выводы ПОСЛЕ выполнения кода.

    Также возвращает рекомендуемые generation_kwargs для Gemini (например, response_mime_type).
    """

    ALLOWED_LABELS = (
        "info",
        "n_code",
        "r_code",
        "math",
        "conclusion",
        "code_conclusion",
    )

    def __init__(
        self,
        code_mime: str = "text/plain",
    ):
        self.code_mime = code_mime
        self.logger = logging.getLogger(f"{__name__}.TaskPrompter")

    def compose(
        self,
        base_user_text: str,
        label: str,
        need_conclusion: bool = False,
        extra_hints: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Возвращает:
          - готовый текст промпта (base_user_text + suffix),
          - рекомендуемые generation_kwargs (можно передать в GeminiClient.send_message).

        Args:
            base_user_text: исходный текст задачи
            label: тип сегмента ("info" | "n_code" | "r_code" | "math" | "conclusion" | "code_conclusion")
            need_conclusion: устаревший флаг; для code-задач выводы теперь формируются отдельным шагом
            extra_hints: дополнительные указания (опционально)

        Returns:
            (final_text, generation_kwargs)
        """
        if label not in self.ALLOWED_LABELS:
            raise ValueError(f"Недопустимый label: {label}")

        suffix = self._suffix_for(
            label=label, need_conclusion=need_conclusion, extra_hints=extra_hints
        )
        final_text = self._join_tail(base_user_text, suffix)
        gen_kwargs = self._generation_kwargs_for(label=label)

        self.logger.debug(
            f"Сформирован промпт для '{label}'. Длина base={len(base_user_text)}, tail={len(suffix)}."
        )
        return final_text, gen_kwargs

    def _join_tail(self, base: str, tail: str) -> str:
        if not base:
            return tail
        if not tail:
            return base
        return f"{base.rstrip()}\n\n{tail.strip()}\n"

    def _generation_kwargs_for(self, label: str) -> Dict[str, Any]:
        """
        Рекомендуемые параметры генерации.
        Для кода просим «чистый текст», без Markdown; модель вернёт просто код.
        """
        if label in ("n_code", "r_code"):
            return {"response_mime_type": self.code_mime}
        return {"response_mime_type": "text/plain"}

    def _suffix_for(
        self, label: str, need_conclusion: bool, extra_hints: Optional[str]
    ) -> str:
        if label == "n_code":
            return self._suffix_n_code(need_conclusion, extra_hints)
        if label == "r_code":
            return self._suffix_r_code(need_conclusion, extra_hints)
        if label == "math":
            return self._suffix_math(extra_hints)
        if label == "conclusion":
            return self._suffix_conclusion(extra_hints)
        if label == "code_conclusion":
            return self._suffix_code_conclusion(extra_hints)
        if label == "info":
            return self._suffix_info(extra_hints)
        return ""

    def _suffix_info(self, extra: Optional[str]) -> str:
        return ""

    def _suffix_n_code(self, need_conclusion: bool, extra: Optional[str]) -> str:
        base = """
Верни только исполняемый код одной ячейки.
Требования к выводу:
- Без Markdown, без тройных кавычек и бэктиков, без пояснений до/после.
- Импорты указывай только те, которых ещё не было в контексте. Код, который был в контексте, не повторяй. Не переписывай ячейки, которые были выше.
- Пиши простой код. Используй принципы YAGNI и KISS. Важно: Не пиши комментарии в коде.
- Никаких текстовых выводов — только исполняемый код.
- Проверь, что код точно выполняет все требования.
"""
        return base + (f"\nДополнительно: {extra}\n" if extra else "")

    def _suffix_r_code(self, need_conclusion: bool, extra: Optional[str]) -> str:
        base = """
Исправь/дополни существующий код.
Требования к выводу:
- Верни полный финальный код ячейки (не diff и не патч), без Markdown и без тройных кавычек.
- Сохраняй структуру и имена переменных, меняй минимально, только по сути. 
- Импорты указывай только те, которых ещё не было в контексте. Код, который был в контексте, не повторяй.
- Не добавляй пояснений о том, что было изменено.
- Пиши простой код. Используй принципы YAGNI и KISS. Важно: Не пиши комментарии в коде.
- Однако ты должен сохранить все комментарии и докстринги, которые уже есть в коде, важно не добавлять свои.
- Никаких текстовых выводов — только код.
- Проверь, что код точно выполняет все требования.
"""
        return base + (f"\nДополнительно: {extra}\n" if extra else "")

    def _suffix_math(self, extra: Optional[str]) -> str:
        base = r"""
Дай математическое решение на русском.
пиши в стиле katex(формулы выделяй в $) Это нужно чтобы они правильно отображались, а не как код.
Пример: $\phi+\xi$ - не отделяй $ от формулы пробелами.
Придерживайся стиля 'чистая математическая выкладка'.
Твой ответ — это только цепочка формул, структурированная отступами и переносами строк. 
Полностью исключи повествовательный текст, слова-заголовки, списки и любые комментарии. 
Для определений используй :=, для логических следствий — =>.
"""
        return base + (f"\nДополнительно: {extra}\n" if extra else "")

    def _suffix_conclusion(self, extra: Optional[str]) -> str:
        base = """
Требуется ответить на вопрос или написать вывод.
Пиши как человек. Избегай лишней структурированности. Пиши простыми предложениями. Используй меньше речевых оборотов. Словарный запас должен быть скудным. Меньше эмоций. Меньше воды.
"""
        return base + (f"\nДополнительно: {extra}\n" if extra else "")

    def _suffix_code_conclusion(self, extra: Optional[str]) -> str:
        base = """
Сформулируй выводы по результатам выполнения предыдущей ячейки.
Пиши как человек. Избегай лишней структурированности. Пиши простыми предложениями. Используй меньше речевых оборотов. Словарный запас должен быть скудным. Меньше эмоций. Меньше воды.
"""
        return base + (f"\nДополнительно: {extra}\n" if extra else "")
