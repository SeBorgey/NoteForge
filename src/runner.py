import os
import time
import base64
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from queue import Empty

from jupyter_client import KernelManager


@dataclass
class ImageData:
    mime_type: str           # "image/png", "image/svg+xml", "image/jpeg"
    data: bytes              # PNG/JPEG: bytes; SVG: utf-8 bytes
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    result_text: Optional[str]              # repr последнего выражения (если было)
    images: List[ImageData]                 # картинки этого запуска
    rich: List[Dict[str, Any]]              # сырой rich-вывод (data/metadata)
    error: Optional[Dict[str, Any]]         # {"ename","evalue","traceback":[...]} или None
    elapsed_sec: float


class CodeRunner:
    """
    Синхронный запуск кода в Jupyter kernel с захватом stdout/stderr/rich-выводов (включая картинки).
    Single Responsibility: только работа с kernel и сбор артефактов.
    """
    def __init__(
        self,
        kernel_name: str = "python3",
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        startup_timeout: float = 30.0,
        execution_timeout: float = 1800.0,  # 30 минут по умолчанию
        preserve_state: bool = True,
        prelude_code: Optional[str] = "%matplotlib inline\n",
        log_level: int = logging.INFO,
    ):
        self.kernel_name = kernel_name
        self.working_dir = working_dir or os.getcwd()
        self.env = dict(os.environ, **(env or {}))
        # Inline-бэкенд для Matplotlib
        self.env.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")

        self.startup_timeout = startup_timeout
        self.execution_timeout = execution_timeout
        self.preserve_state = preserve_state
        self.prelude_code = prelude_code or ""
        self.logger = logging.getLogger(f"{__name__}.CodeRunner")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.km: Optional[KernelManager] = None
        self.kc = None

    # Контекст-менеджер
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown(now=True)

    # Жизненный цикл ядра
    def start(self):
        if self.km is not None:
            self.logger.debug("Ядро уже запущено.")
            return

        self.logger.info("🚀 Старт Jupyter kernel...")
        self.km = KernelManager(kernel_name=self.kernel_name)
        self.km.start_kernel(cwd=self.working_dir, env=self.env)
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=self.startup_timeout)
        self.logger.info("✓ Ядро готово.")

        if self.prelude_code.strip():
            self.logger.debug("Выполняем prelude_code (silent)...")
            self._execute_internal(self.prelude_code, silent=True, timeout=self.execution_timeout)

    def shutdown(self, now: bool = True):
        if self.kc is not None:
            self.logger.info("⏹ Остановка каналов kernel client...")
            self.kc.stop_channels()
            self.kc = None
        if self.km is not None:
            self.logger.info("🛑 Завершение kernel...")
            try:
                self.km.shutdown_kernel(now=now)
            except Exception as e:
                self.logger.warning(f"Не удалось корректно завершить ядро: {e}")
            self.km = None

    def restart(self):
        if self.km is None:
            self.start()
            return

        self.logger.info("🔁 Перезапуск ядра...")
        try:
            # 1) Обычный restart
            self.km.restart_kernel(now=True)
            self.kc = self.km.client()
            self.kc.start_channels()
            self.kc.wait_for_ready(timeout=self.startup_timeout)
            self.logger.info("✓ Ядро перезапущено.")
        except Exception as e:
            self.logger.warning(f"Перезапуск ядра не удался: {e}. Пробуем полный рестарт...")

            # 2) Полный рестарт (shutdown → start)
            try:
                self.shutdown(now=True)  # корректно гасим текущее ядро
            except Exception as e2:
                self.logger.warning(f"Проблема при остановке ядра: {e2}")

            self.km = None
            # Немного подождать, чтобы порт освободился (редко, но помогает)
            time.sleep(1.0)

            # 3) Старт с нуля
            self.start()

        # Prelude после успешного (любого) рестарта
        if self.prelude_code.strip():
            self.logger.debug("Выполняем prelude_code после перезапуска (silent)...")
            self._execute_internal(self.prelude_code, silent=True, timeout=self.execution_timeout)


    # Публичный API
    def execute(self, code: str, timeout: Optional[float] = None) -> ExecutionResult:
        if self.km is None or self.kc is None:
            self.start()
        elif not self.preserve_state:
            self.restart()

        self._drain_iopub()

        effective_timeout = self.execution_timeout if timeout is None else timeout
        self.logger.info("▶ Выполнение кода...")
        self.logger.debug("Код:\n" + code)

        try:
            result = self._execute_internal(code, silent=False, timeout=effective_timeout)
        except TimeoutError as te:
            self.logger.error(f"✗ Таймаут выполнения: {te}")
            raise

        if result.error:
            self.logger.error(f"✗ Ошибка: {result.error.get('ename')}: {result.error.get('evalue')}")
        else:
            self.logger.info(
                f"✓ Готово за {result.elapsed_sec:.2f} c, "
                f"stdout={len(result.stdout)} симв., stderr={len(result.stderr)} симв., "
                f"картинок={len(result.images)}"
            )
        return result

    # Внутреннее исполнение
    def _execute_internal(self, code: str, silent: bool, timeout: Optional[float]) -> ExecutionResult:
        t0 = time.perf_counter()
        msg_id = self.kc.execute(code, silent=silent, store_history=True, allow_stdin=False, stop_on_error=False)

        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        images: List[ImageData] = []
        rich: List[Dict[str, Any]] = []
        result_text: Optional[str] = None
        error: Optional[Dict[str, Any]] = None

        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=0.2)
            except Empty:
                if (timeout is not None) and ((time.perf_counter() - t0) > timeout):
                    self.logger.warning(f"⏳ Таймаут {timeout}s. Прерываем ядро...")
                    try:
                        self.km.interrupt_kernel()
                    except Exception:
                        pass
                    # Пробрасываем исключение наверх — НЕ превращаем это в «ошибку кода» для LLM
                    raise TimeoutError(f"Execution exceeded {timeout} seconds")
                continue

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "status" and content.get("execution_state") == "idle":
                break

            if msg_type == "stream":
                name = content.get("name")
                text = content.get("text", "")
                if name == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)

            elif msg_type in ("display_data", "execute_result"):
                data = content.get("data", {}) or {}
                metadata = content.get("metadata", {}) or {}
                rich.append({"data": data, "metadata": metadata, "msg_type": msg_type})

                if msg_type == "execute_result" and "text/plain" in data:
                    result_text = self._to_text(data["text/plain"])

                for mime in ("image/png", "image/jpeg", "image/svg+xml"):
                    if mime in data:
                        payload = data[mime]
                        if mime == "image/svg+xml":
                            img_bytes = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
                        else:
                            img_bytes = base64.b64decode(payload) if isinstance(payload, str) else bytes(payload)
                        images.append(ImageData(mime_type=mime, data=img_bytes, metadata=metadata))

            elif msg_type == "error":
                error = {
                    "ename": content.get("ename"),
                    "evalue": content.get("evalue"),
                    "traceback": content.get("traceback", []),
                }

        try:
            reply = self.kc.get_shell_msg(timeout=1.0)
            if reply["parent_header"].get("msg_id") == msg_id:
                rc = reply.get("content", {})
                if rc.get("status") == "error" and not error:
                    error = {
                        "ename": rc.get("ename"),
                        "evalue": rc.get("evalue"),
                        "traceback": rc.get("traceback", []),
                    }
        except Empty:
            pass

        elapsed = time.perf_counter() - t0
        return ExecutionResult(
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            result_text=result_text,
            images=images,
            rich=rich,
            error=error,
            elapsed_sec=elapsed,
        )

    def _drain_iopub(self):
        if not self.kc:
            return
        drained = 0
        while True:
            try:
                _ = self.kc.get_iopub_msg(timeout=0.05)
                drained += 1
            except Empty:
                break
        if drained:
            self.logger.debug(f"Очистили {drained} сообщений из IOPub перед запуском.")

    @staticmethod
    def _to_text(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        try:
            return str(x)
        except Exception:
            return repr(x)