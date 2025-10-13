import logging
from pathlib import Path

from client import GeminiClient
from parser import NotebookTaskSplitter
from runner import CodeRunner
from prompter import TaskPrompter
from solver import NotebookSolver
import config

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s | %(levelname)-8s | %(message)s",
datefmt="%H:%M:%S",
)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    API_KEY = config.API
    INPUT_NOTEBOOK = "test.ipynb"
    OUTPUT_NOTEBOOK = "homework_solved.ipynb"

    print("=" * 80)
    print("JUPYTER NOTEBOOK SOLVER with Gemini")
    print("=" * 80)

    gemini = GeminiClient(
        api_key=API_KEY,
        model_name="gemini-2.5-pro",
        requests_per_minute=4,
        max_retries=3
    )

    splitter = NotebookTaskSplitter(
        gemini_client=gemini,
        temperature=0.2,
        top_p=0.1,
    )

    runner = CodeRunner(
        kernel_name="python3",
        startup_timeout=120.0,
        execution_timeout=1800.0,
        preserve_state=True,
        prelude_code="%matplotlib inline\n"
    )


    prompter = TaskPrompter(
    )

    solver = NotebookSolver(
        gemini_client=gemini,
        task_splitter=splitter,
        code_runner=runner,
        prompter=prompter,
        log_level=logging.INFO,
        log_file=None,
        max_code_fix_attempts=3,
        exec_timeout=60.0
    )

    with runner:
        result_path = solver.solve(
            ipynb_path=INPUT_NOTEBOOK,
            output_path=OUTPUT_NOTEBOOK
        )

    print(f"\n✅ Готово! Результат: {result_path}")


if __name__ == "__main__":
    main()
