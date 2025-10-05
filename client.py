import time
import logging
from collections import deque
from typing import Optional, Union, List, Dict
import google.generativeai as genai


class GeminiClient:
    """
    Клиент для работы с Gemini API с поддержкой квот и повторных попыток.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-pro",
        requests_per_minute: int = 5,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        log_level: int = logging.INFO
    ):
        """
        Инициализация клиента Gemini.
        
        Args:
            api_key: API ключ для Gemini
            model_name: Название модели
            requests_per_minute: Максимальное количество запросов в минуту
            max_retries: Максимальное количество попыток при ошибке
            retry_delay: Задержка между повытками в секундах
            log_level: Уровень логирования
        """
        self.api_key = api_key
        self.model_name = model_name
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Настройка логирования
        self.logger = self._setup_logger(log_level)
        
        # Инициализация клиента Gemini
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.logger.info(f"✓ GeminiClient успешно инициализирован")
            self.logger.info(f"  Модель: {self.model_name}")
            self.logger.info(f"  Лимит запросов: {self.requests_per_minute} в минуту")
        except Exception as e:
            self.logger.error(f"✗ Ошибка инициализации Gemini: {e}")
            raise
        
        # Очередь для отслеживания времени запросов
        # Хранит timestamp последних N запросов
        self.request_times = deque(maxlen=self.requests_per_minute)
    
    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Настраивает логгер для класса."""
        logger = logging.getLogger(f"{__name__}.GeminiClient")
        logger.setLevel(log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
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
            message: Текст сообщения или список сообщений для диалога
                    Для диалога передайте историю в формате:
                    [
                        {"role": "user", "parts": ["текст"]},
                        {"role": "model", "parts": ["текст"]},
                        ...
                    ]
            **generation_kwargs: Дополнительные параметры для генерации
                (temperature, max_output_tokens, top_p, top_k и т.д.)
            
        Returns:
            Текст ответа от модели
            
        Raises:
            Exception: Если все попытки запроса завершились неудачей
        """
        # Определяем тип сообщения для логов
        msg_preview = (
            f"{message[:50]}..." if isinstance(message, str) 
            else f"История из {len(message)} сообщений"
        )
        self.logger.info(f"➤ Новый запрос: {msg_preview}")
        
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # Соблюдаем квоту
                self._wait_if_needed()
                
                # Регистрируем запрос (важно: делаем это ДО отправки,
                # т.к. даже неудачная попытка считается за запрос к API)
                self._record_request()
                
                self.logger.info(
                    f"🔄 Попытка {attempt}/{self.max_retries}: отправка запроса..."
                )
                
                # Формируем конфигурацию генерации
                generation_config = None
                if generation_kwargs:
                    generation_config = genai.types.GenerationConfig(
                        **generation_kwargs
                    )
                
                # Отправляем запрос
                if isinstance(message, str):
                    # Простое сообщение
                    response = self.model.generate_content(
                        message,
                        generation_config=generation_config
                    )
                else:
                    # Диалог с историей
                    chat = self.model.start_chat(history=message[:-1])
                    response = chat.send_message(
                        message[-1]["parts"][0],
                        generation_config=generation_config
                    )
                
                # Получаем текст ответа
                response_text = response.text
                
                self.logger.info(
                    f"✓ Ответ получен (длина: {len(response_text)} символов)"
                )
                self.logger.debug(f"Превью ответа: {response_text[:100]}...")
                
                return response_text
            
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                error_msg = str(e)
                
                self.logger.error(
                    f"✗ Ошибка при попытке {attempt}/{self.max_retries}: "
                    f"{error_type}: {error_msg}"
                )
                
                # Если это не последняя попытка - ждём перед повтором
                if attempt < self.max_retries:
                    self.logger.info(
                        f"⏳ Ожидание {self.retry_delay} сек перед повторной попыткой..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    # Все попытки исчерпаны
                    self.logger.error(
                        f"✗ Все {self.max_retries} попытки исчерпаны. Запрос провалился."
                    )
        
        # Если мы здесь, значит все попытки провалились
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


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    # Настройка базового логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ GeminiClient")
    print("=" * 70)
    
    # ВАЖНО: Замените на свой API ключ!
    API_KEY = "your-api-key-here"
    
    # Создаём клиента
    client = GeminiClient(
        api_key=API_KEY,
        model_name="gemini-2.5-pro",
        requests_per_minute=5,
        max_retries=3,
        retry_delay=1.0
    )
    
    print("\n" + "=" * 70)
    print("ПРИМЕР 1: Простой запрос")
    print("=" * 70)
    
    try:
        response = client.send_message("Привет! Напиши короткую шутку про программистов.")
        print(f"\n📝 Ответ:\n{response}\n")
    except Exception as e:
        print(f"❌ Ошибка: {e}\n")
    
    print("=" * 70)
    print("ПРИМЕР 2: Запрос с параметрами генерации")
    print("=" * 70)
    
    try:
        response = client.send_message(
            "Напиши очень короткое стихотворение про Python",
            temperature=0.9,
            max_output_tokens=100
        )
        print(f"\n📝 Ответ:\n{response}\n")
    except Exception as e:
        print(f"❌ Ошибка: {e}\n")
    
    print("=" * 70)
    print("ПРИМЕР 3: Диалог с историей")
    print("=" * 70)
    
    # Формируем историю диалога
    history = [
        {"role": "user", "parts": ["Давай поиграем в игру. Я загадал число от 1 до 10."]},
        {"role": "model", "parts": ["Отлично! Это 5?"]},
        {"role": "user", "parts": ["Нет, больше."]},
    ]
    
    try:
        response = client.send_message(history)
        print(f"\n📝 Ответ:\n{response}\n")
    except Exception as e:
        print(f"❌ Ошибка: {e}\n")
    
    print("=" * 70)
    print("ПРИМЕР 4: Демонстрация работы квоты (6 запросов подряд)")
    print("=" * 70)
    
    for i in range(6):
        try:
            response = client.send_message(f"Скажи просто 'Ответ {i+1}'")
            print(f"✓ Запрос {i+1} выполнен\n")
        except Exception as e:
            print(f"❌ Запрос {i+1} провалился: {e}\n")
    
    print("=" * 70)
    print("Демонстрация завершена!")
    print("=" * 70)