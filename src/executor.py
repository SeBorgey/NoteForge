import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from runner import CodeRunner, ExecutionResult

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class ICodeFixer(Protocol):
    """
    Абстракция исправителя кода.
    Возвращает новый код (str), если предлагает исправление; иначе None.
    Никаких внешних вызовов здесь нет — вы можете реализовать хоть ручной фиксер, хоть эвристический.
    """
    def suggest_fix(self, code: str, error: Optional[Dict[str, Any]], stdout: str, stderr: str) -> Optional[str]:
        ...


class NoOpFixer:
    """Фиксер по умолчанию — ничего не делает, всегда возвращает None."""
    def suggest_fix(self, code: str, error: Optional[Dict[str, Any]], stdout: str, stderr: str) -> Optional[str]:
        return None


@dataclass
class FixAttempt:
    attempt_no: int
    applied: bool
    success: bool
    new_code_preview: str = ""
    error: Optional[Dict[str, Any]] = None


@dataclass
class LastCellRunResult:
    success: bool
    final_code: str
    execution: "ExecutionResult"
    attempts: List[FixAttempt] = field(default_factory=list)


class LastCellExecutor:
    """
    Стратегии подготовки:
      - 'auto': исполняем пролог только если он изменился с прошлого раза
      - 'always': всегда перезапуск ядра и исполнение пролога
      - 'never': считаем, что состояние уже подготовлено
    """
    def __init__(
        self,
        runner: "CodeRunner",
        fixer: Optional[ICodeFixer] = None,
        log_level: int = logging.INFO
    ):
        self.runner = runner
        self.fixer = fixer or NoOpFixer()

        self.logger = logging.getLogger(f"{__name__}.LastCellExecutor")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self._prepared_once = False
        self._preamble_fp: Optional[str] = None

    def run(
        self,
        cells: List[str],
        prepare_strategy: str = "auto",
        max_fixes: int = 0  # по умолчанию без автопочинки
    ) -> LastCellRunResult:
        assert cells, "Нужна хотя бы одна ячейка."
        pre_cells = cells[:-1]
        last_code = cells[-1]

        # Подготовка состояния
        ok = self._prepare_state_if_needed(pre_cells, prepare_strategy)
        if not ok:
            raise RuntimeError("Ошибка в подготовительных ячейках.")

        # Первая попытка последней ячейки
        res = self.runner.execute(last_code)
        attempts: List[FixAttempt] = []
        if not res.error:
            return LastCellRunResult(True, last_code, res, attempts)

        # Попытки фикса (если включено)
        for attempt in range(1, max_fixes + 1):
            self.logger.warning(f"⚠ Ошибка в последней ячейке. Попытка исправления {attempt}/{max_fixes}...")
            proposal = self.fixer.suggest_fix(last_code, res.error, res.stdout, res.stderr)

            if not proposal or proposal.strip() == last_code.strip():
                attempts.append(FixAttempt(attempt_no=attempt, applied=False, success=False, error=res.error))
                self.logger.info("Нет применимого исправления от фиксатора. Останавливаемся.")
                break

            self.logger.info("🔁 Пробуем исправленный код последней ячейки...")
            res = self.runner.execute(proposal)
            attempts.append(FixAttempt(
                attempt_no=attempt,
                applied=True,
                success=not bool(res.error),
                new_code_preview=self._short(proposal),
                error=None if not res.error else res.error
            ))
            last_code = proposal
            if not res.error:
                self.logger.info("✓ Исправление сработало.")
                return LastCellRunResult(True, last_code, res, attempts)

        # Не удалось исправить
        return LastCellRunResult(False, last_code, res, attempts)

    # Внутренняя кухня
    def _prepare_state_if_needed(self, pre_cells: List[str], strategy: str) -> bool:
        if not pre_cells:
            return True

        need_prepare = False
        if strategy == "always":
            need_prepare = True
        elif strategy == "never":
            need_prepare = False
        elif strategy == "auto":
            fp = self._fingerprint(pre_cells)
            need_prepare = (not self._prepared_once) or (self._preamble_fp != fp)
            self._preamble_fp = fp
        else:
            raise ValueError("prepare_strategy должен быть 'auto'|'always'|'never'.")

        if not need_prepare:
            self.logger.info("⏭ Пролог не изменился (‘auto’/’never’) — не переисполняем.")
            return True

        self.logger.info("🔁 Перезапуск ядра и исполнение пролога...")
        self.runner.restart()
        for idx, code in enumerate(pre_cells):
            self.logger.info(f"▶ Ячейка пролога #{idx}...")
            r = self.runner.execute(code)
            if r.error:
                self.logger.error(f"✗ Ошибка в прологе #{idx}: {r.error.get('ename')}: {r.error.get('evalue')}")
                return False

        self._prepared_once = True
        self.logger.info("✓ Состояние подготовлено.")
        return True

    @staticmethod
    def _fingerprint(codes: List[str]) -> str:
        joined = "\n# ----- CELL SPLIT -----\n".join(codes)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    @staticmethod
    def _short(s: str, limit: int = 240) -> str:
        s = (s or "").replace("\n", "\\n")
        return s if len(s) <= limit else s[: limit - 3] + "..."


