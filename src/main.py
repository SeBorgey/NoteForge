import logging
from pathlib import Path

from client import GeminiClient
from parser import NotebookTaskSplitter
from runner import CodeRunner
from prompter import TaskPrompter
from solver import NotebookSolver
import config

def main():
    # Настройка базового логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    # Параметры
    API_KEY = config.API
    INPUT_NOTEBOOK = "test.ipynb"
    OUTPUT_NOTEBOOK = "homework_solved.ipynb"

    print("=" * 80)
    print("JUPYTER NOTEBOOK SOLVER with Gemini")
    print("=" * 80)

    # 1. Создаём клиент Gemini
    gemini = GeminiClient(
        api_key=API_KEY,
        model_name="gemini-2.5-pro",
        requests_per_minute=2,
        max_retries=3
    )

    # 2. Создаём разметчик задач
    splitter = NotebookTaskSplitter(
        gemini_client=gemini,
        temperature=0.2,
        top_p=0.1,
    )

    # 3. Создаём исполнитель кода
    runner = CodeRunner(
        kernel_name="python3",
        startup_timeout=120.0,
        execution_timeout=1800.0,
        preserve_state=True,
        prelude_code="%matplotlib inline\n"
    )

    # 4. Создаём генератор промптов
    prompter = TaskPrompter(
        code_language="Python"
    )

    # 5. Создаём главный solver
    solver = NotebookSolver(
        gemini_client=gemini,
        task_splitter=splitter,
        code_runner=runner,
        prompter=prompter,
        log_level=logging.INFO,
        log_file=None,          # убрали лог-файл
        max_code_fix_attempts=3,
        exec_timeout=60.0
    )

    # 6. Запуск
    with runner:
        result_path = solver.solve(
            ipynb_path=INPUT_NOTEBOOK,
            output_path=OUTPUT_NOTEBOOK
        )

    print(f"\n✅ Готово! Результат: {result_path}")


if __name__ == "__main__":
    main()
