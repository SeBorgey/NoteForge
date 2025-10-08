import time
import logging
from collections import deque
from typing import Optional, Union, List, Dict
import google.generativeai as genai
import json



class GeminiClient:
    """
    Клиент для работы с Gemini API с поддержкой квот и повторных попыток.
    """
    
    def __init__(
        self,
        api_key: Union[str, List[str]],
        model_name: str = "gemini-2.5-pro",
        requests_per_minute: int = 5,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        log_level: int = logging.INFO
    ):
        """
        Инициализация клиента Gemini.
        
        Args:
            api_key: API ключ для Gemini или список ключей для чередования
            model_name: Название модели
            requests_per_minute: Максимальное количество запросов в минуту
            max_retries: Максимальное количество попыток при ошибке
            retry_delay: Задержка между повытками в секундах
            log_level: Уровень логирования
        """
        # Обработка одного ключа или списка ключей
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        else:
            self.api_keys = api_key
        
        self.current_key_index = 0
        self.model_name = model_name
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Настройка логирования
        self.logger = self._setup_logger(log_level)
        
        # Инициализация клиента Gemini
        try:
            genai.configure(api_key=self.api_keys[self.current_key_index])
            self.model = genai.GenerativeModel(self.model_name)
            self.logger.info(f"✓ GeminiClient успешно инициализирован")
            self.logger.info(f"  Модель: {self.model_name}")
            self.logger.info(f"  Количество API ключей: {len(self.api_keys)}")
            self.logger.info(f"  Лимит запросов: {self.requests_per_minute} в минуту")
        except Exception as e:
            self.logger.error(f"✗ Ошибка инициализации Gemini: {e}")
            raise
        
        # Очередь для отслеживания времени запросов
        # Хранит timestamp последних N запросов
        self.request_times = deque(maxlen=self.requests_per_minute)

    def _extract_response_text(self, response) -> str:
        # 1) Самый простой путь
        try:
            t = response.text
            if t and t.strip():
                return t
        except Exception:
            pass

        # 2) Собрать текст из candidates/parts
        candidates = getattr(response, "candidates", None) or []
        collected = []
        finish_reasons = []
        for c in candidates:
            fr = getattr(c, "finish_reason", None)
            finish_reasons.append(str(fr))
            content = getattr(c, "content", None)
            parts = getattr(content, "parts", None) or []
            for p in parts:
                txt = getattr(p, "text", None)
                if txt:
                    collected.append(txt)

        if collected:
            return "\n".join(collected).strip()

        # 3) Диагностика (например, SAFETY/BLOCKLIST или пустой вывод)
        pf = getattr(response, "prompt_feedback", None)
        block_reason = getattr(pf, "block_reason", None) if pf else None
        safety = getattr(pf, "safety_ratings", None) if pf else None
        raise RuntimeError(
            f"Пустой ответ модели. finish_reasons={finish_reasons}, "
            f"block_reason={block_reason}, safety={safety}"
        )

    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Настраивает логгер для класса."""
        logger = logging.getLogger(f"{__name__}.GeminiClient")
        return logger
    
    def _switch_api_key(self):
        """Переключает API ключ на следующий в списке."""
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            genai.configure(api_key=self.api_keys[self.current_key_index])
            self.model = genai.GenerativeModel(self.model_name)  # Пересоздаём модель!
            self.logger.debug(f"Переключение на API ключ #{self.current_key_index + 1}")
    
    def _wait_if_needed(self):
        """
        Проверяет квоту и при необходимости ждёт.
        Логика: если сделано N запросов, то (N+1)-й запрос должен ждать,
        пока с момента первого не пройдёт минута.
        """
        current_time = time.time()
        
        # Если очередь ещё не заполнена, можно делать запрос сразу
        if len(self.request_times) < self.requests_per_minute:
            self.logger.debug(
                f"Запросов в окне: {len(self.request_times)}/{self.requests_per_minute}. "
                "Ожидание не требуется."
            )
            return
        
        # Очередь заполнена - проверяем, прошла ли минута с самого старого запроса
        oldest_request_time = self.request_times[0]
        time_since_oldest = current_time - oldest_request_time
        
        if time_since_oldest < 60:
            wait_time = 60 - time_since_oldest
            self.logger.warning(
                f"⏳ Достигнут лимит запросов ({self.requests_per_minute}/мин). "
                f"Ожидание {wait_time:.1f} сек..."
            )
            time.sleep(wait_time)
            self.logger.info("✓ Ожидание завершено")
    
    def _record_request(self):
        """Записывает время выполнения запроса."""
        self.request_times.append(time.time())
        self.logger.debug(
            f"Запрос зарегистрирован. "
            f"Запросов в текущем окне: {len(self.request_times)}/{self.requests_per_minute}"
        )
    
    def send_message(
        self,
        message: Union[str, List[Dict]],
        **generation_kwargs
    ) -> str:
        """
        Отправляет сообщение в Gemini и возвращает ответ.

        Args:
            message: Текст сообщения или список сообщений для диалога.
                    Поддерживаются parts с inline_data для изображений.
            **generation_kwargs: Параметры генерации (temperature, max_output_tokens, top_p, top_k, response_mime_type и т.д.)

        Returns:
            Текст ответа от модели

        Raises:
            Exception: Если все попытки запроса завершились неудачей
        """
        # Переключаем API ключ перед каждым запросом
        self._switch_api_key()
        
        # Полный вывод запроса в лог (без обрезки)
        def _format_payload(msg: Union[str, List[Dict]]) -> str:
            if isinstance(msg, str):
                return msg
            try:
                return json.dumps(msg, ensure_ascii=False, indent=2)
            except Exception:
                return str(msg)

        payload_str = _format_payload(message)
        self.logger.info("➤ Новый запрос к LLM (полностью):\n%s", payload_str)

        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # Квота
                self._wait_if_needed()
                # Регистрируем запрос (попытка тоже учитывается)
                self._record_request()

                self.logger.info("🔄 Попытка %d/%d: отправка...", attempt, self.max_retries)

                generation_config = None
                if generation_kwargs:
                    generation_config = genai.types.GenerationConfig(**generation_kwargs)

                # Отправка
                if isinstance(message, str):
                    response = self.model.generate_content(
                        message,
                        generation_config=generation_config
                    )
                else:
                    response = self.model.generate_content(
                        message,
                        generation_config=generation_config
                    )

                # Текст ответа
                response_text = self._extract_response_text(response)

                # Полный вывод ответа в лог (без обрезки)
                self.logger.info("⬅ Ответ модели (полностью, %d симв.):\n%s", len(response_text), response_text)

                return response_text

            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                error_msg = str(e)
                self.logger.error("✗ Ошибка при попытке %d/%d: %s: %s", attempt, self.max_retries, error_type, error_msg)

                if attempt < self.max_retries:
                    self.logger.info("⏳ Ждём %.1f сек перед повторной попыткой...", self.retry_delay)
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error("✗ Все %d попытки исчерпаны.", self.max_retries)

        raise Exception(
            f"Не удалось выполнить запрос после {self.max_retries} попыток. "
            f"Последняя ошибка: {type(last_exception).__name__}: {str(last_exception)}"
        )
    
    def set_quota(self, requests_per_minute: int):
        """
        Изменяет квоту запросов в минуту.
        
        Args:
            requests_per_minute: Новое значение квоты
        """
        old_quota = self.requests_per_minute
        self.requests_per_minute = requests_per_minute
        self.request_times = deque(
            list(self.request_times)[-requests_per_minute:],
            maxlen=requests_per_minute
        )
        self.logger.info(
            f"Квота изменена: {old_quota} → {requests_per_minute} запросов/мин"
        )