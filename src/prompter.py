import logging
from typing import Dict, Optional, Any, Tuple


class TaskPrompter:
    """
    Формирует «приписку» (tail/suffix) к пользовательскому сообщению в зависимости от типа задачи.
    Это НЕ системный промпт — просто добавка к концу prompt'а.

    Идея:
      - Для code-задач (n_code/r_code) возвращаем жёсткие указания: «только код», без Markdown/объяснений,
        стабильные импорты и т.п. Если нужно сделать выводы — в виде Python-комментариев ('# Вывод: ...').
      - Для math — пошаговое решение + 'Итог:' в конце.
      - Для conclusion — краткий финальный ответ, 1–3 предложения/пункта.
      - Для info — коротко, без кода, по делу.

    Также можно вернуть рекомендуемые generation_kwargs для Gemini (например, response_mime_type).

    Пример использования:
        prompter = TaskPrompter()
        text, gen_kwargs = prompter.compose(
            base_user_text="Напиши функцию, которая считает среднее по списку",
            label="n_code",
            need_conclusion=True
        )
        # далее вы отправляете text в модель, а gen_kwargs передаёте как **generation_kwargs
    """

    ALLOWED_LABELS = ("info", "n_code", "r_code", "math", "conclusion")

    def __init__(
        self,
        code_language: str = "Python",
        code_mime: str = "text/plain",  # безопасно для Gemini; "application/json" уже используется в другом месте
        log_level: int = logging.INFO
    ):
        self.code_language = code_language
        self.code_mime = code_mime
        self.logger = logging.getLogger(f"{__name__}.TaskPrompter")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # Публичный API

    def compose(
        self,
        base_user_text: str,
        label: str,
        need_conclusion: bool = False,
        extra_hints: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Возвращает:
          - готовый текст промпта (base_user_text + suffix),
          - рекомендуемые generation_kwargs (можно передать в GeminiClient.send_message).

        Args:
            base_user_text: исходный текст задачи
            label: тип сегмента ("info" | "n_code" | "r_code" | "math" | "conclusion")
            need_conclusion: для code-задач — нужно ли добавить краткие выводы (как комментарии)
            extra_hints: дополнительные указания (опционально)

        Returns:
            (final_text, generation_kwargs)
        """
        if label not in self.ALLOWED_LABELS:
            raise ValueError(f"Недопустимый label: {label}")

        suffix = self._suffix_for(label=label, need_conclusion=need_conclusion, extra_hints=extra_hints)
        final_text = self._join_tail(base_user_text, suffix)
        gen_kwargs = self._generation_kwargs_for(label=label)

        self.logger.debug(f"Сформирован промпт для '{label}'. Длина base={len(base_user_text)}, tail={len(suffix)}.")
        return final_text, gen_kwargs

    # Внутренняя кухня

    def _join_tail(self, base: str, tail: str) -> str:
        if not base:
            return tail
        if not tail:
            return base
        # Простой читаемый разделитель, чтобы «приписка» была явно отделена
        return f"{base.rstrip()}\n\n--- СТИЛЬ ОТВЕТА ---\n{tail.strip()}\n"

    def _generation_kwargs_for(self, label: str) -> Dict[str, Any]:
        """
        Рекомендуемые параметры генерации.
        Для кода просим «чистый текст», без Markdown; модель вернёт просто код.
        """
        if label in ("n_code", "r_code"):
            return {
                "response_mime_type": self.code_mime  # "text/plain" — модель вернёт просто текст (код)
            }
        # Для прочих — обычный plain
        return {
            "response_mime_type": "text/plain"
        }

    def _suffix_for(self, label: str, need_conclusion: bool, extra_hints: Optional[str]) -> str:
        if label == "n_code":
            return self._suffix_n_code(need_conclusion, extra_hints)
        if label == "r_code":
            return self._suffix_r_code(need_conclusion, extra_hints)
        if label == "math":
            return self._suffix_math(extra_hints)
        if label == "conclusion":
            return self._suffix_conclusion(extra_hints)
        if label == "info":
            return self._suffix_info(extra_hints)
        return ""

    # Конкретные стили

    def _suffix_info(self, extra: Optional[str]) -> str:
        base = """
Отвечай кратко и по делу.
- Не пиши код, если это явно не требуется.
- Используй короткие абзацы и маркированные списки при необходимости.
- Избегай повторов и общих фраз.
- Не добавляй преамбул вроде «конечно» или «вот».
"""
        return base + (f"\nДополнительно: {extra}\n" if extra else "")

    def _suffix_n_code(self, need_conclusion: bool, extra: Optional[str]) -> str:
        concl = (
            "- В конце добавь краткие выводы в виде Python-комментариев под заголовком '# Вывод:', 1–3 строки.\n"
            if need_conclusion else
            "- Никаких текстовых выводов — только исполняемый код.\n"
        )
        base = f"""
Верни только исполняемый {self.code_language}-код одной ячейки.
Требования к выводу:
- Без Markdown, без тройных кавычек и бэктиков, без пояснений до/после.
- Импорты явно в начале; код должен быть идемпотентным (перезапускаемым).
- Не запрашивай ввод у пользователя; не обращайся к внешним ресурсам без явной инструкции.
- При визуализации используй стандартные средства (например, matplotlib inline), не сохраняй в файл.
- Минимизируй побочные эффекты; фиксируй сид случайности при необходимости (например, np.random.seed(0)).
{concl}"""
        return base + (f"\nДополнительно: {extra}\n" if extra else "")

    def _suffix_r_code(self, need_conclusion: bool, extra: Optional[str]) -> str:
        concl = (
            "- В конце добавь краткие выводы в виде Python-комментариев под заголовком '# Вывод:', 1–3 строки.\n"
            if need_conclusion else
            "- Никаких текстовых выводов — только код.\n"
        )
        base = f"""
Исправь/дополни существующий {self.code_language}-код.
Требования к выводу:
- Верни полный финальный код ячейки (не diff и не патч), без Markdown и без тройных кавычек.
- Сохраняй структуру и имена переменных, меняй минимально, только по сути.
- Ячейка должна быть исполняемой и идемпотентной; импорты — в начале.
- Не добавляй пояснений о том, что было изменено.
{concl}"""
        return base + (f"\nДополнительно: {extra}\n" if extra else "")

    def _suffix_math(self, extra: Optional[str]) -> str:
        base = """
Дай пошаговое математическое решение на русском.
- Коротко комментируй переходы; при необходимости используй LaTeX-подобную запись формул.
- Проверь граничные случаи/допущения.
- В конце отдельной строкой укажи 'Итог: <краткий ответ>'.
"""
        return base + (f"\nДополнительно: {extra}\n" if extra else "")

    def _suffix_conclusion(self, extra: Optional[str]) -> str:
        base = """
Дай окончательный вывод по задаче.
- Кратко: 1–3 предложения или маркированный список до 5 пунктов.
- Без промежуточных рассуждений и пояснений.
- Если требуется число — укажи единицы измерения и разумную точность.
"""
        return base + (f"\nДополнительно: {extra}\n" if extra else "")