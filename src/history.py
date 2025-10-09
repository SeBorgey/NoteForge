import logging
import uuid
import base64
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from runner import ExecutionResult


@dataclass
class _Msg:
    role: str
    parts: List[Any]
    meta: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class _Branch:
    anchor_idx: int
    messages: List[_Msg] = field(default_factory=list)


class ConversationHistory:
    """
    История сообщений с поддержкой «веток ошибок».
      - В обычном режиме все сообщения пишутся в committed (линейная история).
      - Когда код дал ошибку и начались попытки исправления, открываем ветку (open_branch)
        с якорем на «последнее пользовательское сообщение, запустившее код».
        Все последующие user/assistant ходы (включая ошибку, просьбы исправить и т.д.) пишутся в эту ветку.
      - После успешной починки делаем commit_branch(squash=True):
        удаляем все промежуточные ходы от якоря и оставляем ОДИН финальный ответ ассистента,
        как будто он сразу ответил правильно.
      - Пока ветка открыта, модель получает ПОЛНЫЙ контекст: committed + branch (ветка не урезается).

    Формат для Gemini:
      [{"role": "user"|"model", "parts": [ <строки и/или dict-части> ]}, ...]
    Пример части с картинкой:
      {"inline_data": {"mime_type": "image/png", "data": "<base64>"}}
    """

    ALLOWED_ROLES = ("user", "model")

    def __init__(self, log_level: int = logging.INFO):
        self._committed: List[_Msg] = []
        self._branch: Optional[_Branch] = None

        self.logger = logging.getLogger(f"{__name__}.ConversationHistory")

    def add_user(self, text: str, meta: Optional[Dict[str, Any]] = None) -> str:
        return self._add("user", text, meta)

    def add_model(self, text: str, meta: Optional[Dict[str, Any]] = None) -> str:
        return self._add("model", text, meta)


    def add_user_parts(self, parts: List[Any], meta: Optional[Dict[str, Any]] = None) -> str:
        return self._add_parts("user", parts, meta)

    def add_model_parts(self, parts: List[Any], meta: Optional[Dict[str, Any]] = None) -> str:
        return self._add_parts("model", parts, meta)


    def _add(self, role: str, text: str, meta: Optional[Dict[str, Any]]) -> str:
        if role not in self.ALLOWED_ROLES:
            raise ValueError("role должен быть 'user' или 'model'.")
        msg = _Msg(role=role, parts=[text], meta=(meta or {}))
        if self._branch:
            self._branch.messages.append(msg)
        else:
            self._committed.append(msg)
        return msg.id

    def _add_parts(self, role: str, parts: List[Any], meta: Optional[Dict[str, Any]]) -> str:
        if role not in self.ALLOWED_ROLES:
            raise ValueError("role должен быть 'user' или 'model'.")
        if not isinstance(parts, list) or not parts:
            raise ValueError("parts должен быть непустым списком.")
        msg = _Msg(role=role, parts=list(parts), meta=(meta or {}))
        if self._branch:
            self._branch.messages.append(msg)
        else:
            self._committed.append(msg)
        return msg.id


    def open_branch(
        self, anchor: Optional[int] = None, anchor_role: str = "user"
    ) -> None:
        """
        Начинает ветку исправлений.
        anchor:
          - None -> искать последний индекс anchor_role в committed
          - int  -> явный индекс в committed
        Обычно anchor_role="user" и якорем будет последняя пользовательская задача (перед ответом ассистента).
        """
        if self._branch is not None:
            raise RuntimeError("Ветка уже открыта. Нельзя открыть ещё одну.")

        if anchor is None:
            idx = self._find_last_index(role=anchor_role)
            if idx is None:
                raise RuntimeError(
                    f"Не найдено сообщение с ролью '{anchor_role}' для якоря."
                )
            anchor = idx

        if not (0 <= anchor < len(self._committed)):
            raise IndexError("anchor вне диапазона committed.")

        self._branch = _Branch(anchor_idx=anchor, messages=[])
        self.logger.info(
            f"Открыта ветка исправлений на якоре #{anchor} ({anchor_role})."
        )

    def discard_branch(self) -> None:
        """Отменяет ветку, ничего не меняя в committed."""
        if self._branch:
            self.logger.info("Ветка отменена. Промежуточные сообщения удалены.")
        self._branch = None

    def commit_branch(
        self, squash: bool = True, keep_last_exec_summary: bool = True
    ) -> None:
        """
        Завершает ветку.
        - Если squash=True: удаляет всё от якоря и оставляет один финальный ответ ассистента
          (как будто он сразу был дан на якорный user-запрос).
          Дополнительно (если keep_last_exec_summary=True) старается сохранить последнюю сводку выполнения.
        - Если squash=False: разворачивает все сообщения ветки «как есть» после якоря.

        Предполагается, что финальное сообщение ветки — ответ ассистента.
        """
        if not self._branch:
            self.logger.debug("Нет активной ветки для коммита.")
            return

        br = self._branch
        self._branch = None

        if not br.messages:
            self.logger.info("Ветка пуста. Нечего коммитить.")
            return

        new_history = self._committed[: br.anchor_idx + 1]

        if not squash:
            new_history.extend(br.messages)
            self._committed = new_history
            self.logger.info("Ветка коммичена без squash (развёрнута целиком).")
            return

        last_assistant = self._find_last_in_iter(br.messages, role="model")
        if last_assistant is None:
            raise RuntimeError(
                "Нельзя сделать squash: нет финального ответа ассистента в ветке."
            )

        new_history.append(last_assistant)

        if keep_last_exec_summary:
            last_exec = self._find_last_exec_summary(br.messages)
            if last_exec:
                new_history.append(last_exec)

        self._committed = new_history
        self.logger.info(
            "Ветка коммичена со squash: оставлен один финальный ответ ассистента."
        )


    def get_history(self, include_branch: bool = True) -> List[Dict[str, Any]]:
        """
        Возвращает историю в формате Gemini:
          [
            {"role": "user"|"model", "parts": [<str | dict-часть>, ...]},
            ...
          ]
        Если include_branch=True и ветка открыта — добавляет её сообщения в конец.
        """
        msgs: List[_Msg] = list(self._committed)
        if include_branch and self._branch:
            msgs.extend(self._branch.messages)
        return [{"role": m.role, "parts": list(m.parts)} for m in msgs]

    def with_next_user(
        self, text: str, meta: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Удобный метод для отправки запроса в модель: берёт полную историю (включая ветку)
        и добавляет в конец новый user-месседж (НЕ записывая его в историю).
        """
        tmp = self.get_history(include_branch=True)
        tmp.append({"role": "user", "parts": [text]})
        return tmp

    def with_next_user_parts(
        self, parts: List[Any], meta: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Вариант с произвольными parts (строки + inline_data).
        Полезно, когда нужно сразу приложить, например, картинки.
        """
        if not isinstance(parts, list) or not parts:
            raise ValueError("parts должен быть непустым списком.")
        tmp = self.get_history(include_branch=True)
        tmp.append({"role": "user", "parts": list(parts)})
        return tmp

    def clear(self) -> None:
        """Полная очистка истории и ветки."""
        self._committed.clear()
        self._branch = None

    def __len__(self) -> int:
        return len(self._committed) + (len(self._branch.messages) if self._branch else 0)


    def add_execution_result(
        self,
        exec_res: ExecutionResult,
        cell_index: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
        max_field_len: int = 1200,
        attach_images: bool = False,
        max_images: int = 3,
        allowed_mimes: Optional[List[str]] = None,
    ) -> str:
        """
        Добавляет в историю сводку выполнения ячейки.
        По умолчанию добавляет только текстовую сводку. Если attach_images=True —
        прикладывает до max_images картинок в виде inline_data (image/png, image/jpeg).

        Возвращает id добавленного сообщения.
        """
        allowed_mimes = allowed_mimes or ["image/png", "image/jpeg"]

        t_lines = []
        header = "[Выполнение ячейки"
        if cell_index is not None:
            header += f" #{cell_index}"
        header += "]"
        t_lines.append(header)
        t_lines.append(f"elapsed: {exec_res.elapsed_sec:.2f}s")
        if exec_res.error:
            e = exec_res.error
            t_lines.append(f"Ошибка: {e.get('ename')}: {e.get('evalue')}")
            tb = "\n".join(e.get("traceback", []) or [])
            if tb:
                t_lines.append("Traceback (последние строки):")
                t_lines.append(self._truncate(tb, max_field_len))
        else:
            if exec_res.stdout:
                t_lines.append("stdout:")
                t_lines.append(self._truncate(exec_res.stdout, max_field_len))
            if exec_res.stderr:
                t_lines.append("stderr:")
                t_lines.append(self._truncate(exec_res.stderr, max_field_len))
            if exec_res.result_text:
                t_lines.append("repr последнего выражения:")
                t_lines.append(self._truncate(exec_res.result_text, max_field_len))
            if exec_res.images:
                t_lines.append(f"изображений: {len(exec_res.images)}")

        text = "\n".join(t_lines)
        m = dict(meta or {})
        m.setdefault("kind", "exec-result")
        if cell_index is not None:
            m["cell_index"] = cell_index

        if not attach_images:
            return self.add_user(text, meta=m)

        parts: List[Any] = [text]
        attached = 0
        for img in (exec_res.images or []):
            if attached >= max_images:
                break
            if img.mime_type not in allowed_mimes:
                continue
            try:
                b64 = base64.b64encode(img.data).decode("ascii")
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": img.mime_type,
                            "data": b64,
                        }
                    }
                )
                attached += 1
            except Exception as e:
                self.logger.debug(f"Не удалось прикрепить изображение ({img.mime_type}): {e}")

        m["images_total"] = len(exec_res.images or [])
        m["images_attached"] = attached
        return self.add_user_parts(parts, meta=m)


    def _find_last_index(self, role: Optional[str] = None) -> Optional[int]:
        if role is None:
            return len(self._committed) - 1 if self._committed else None
        for i in range(len(self._committed) - 1, -1, -1):
            if self._committed[i].role == role:
                return i
        return None

    @staticmethod
    def _find_last_in_iter(msgs: Iterable[_Msg], role: str) -> Optional[_Msg]:
        for m in reversed(list(msgs)):
            if m.role == role:
                return m
        return None

    @staticmethod
    def _find_last_exec_summary(msgs: Iterable[_Msg]) -> Optional[_Msg]:
        for m in reversed(list(msgs)):
            if m.meta.get("kind") == "exec-result":
                return m
        return None

    @staticmethod
    def _truncate(s: str, limit: int) -> str:
        s = s or ""
        return s if len(s) <= limit else (s[: limit - 3] + "...")