"""支持使用 ``python -m customer_care.evaluation`` 运行评测。"""

from customer_care.evaluation.runner import main
from customer_care.evaluation.trajectory import evaluate_trajectory, format_trajectory_report
from customer_care.evaluation.orchestration import main as orchestration_main


if __name__ == "__main__":
    main()
    print("\n" + format_trajectory_report(evaluate_trajectory()))
    print()
    orchestration_main()
