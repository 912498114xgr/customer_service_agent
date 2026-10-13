from customer_care.evaluation.runner import evaluate_dataset, evaluate_route_policy, load_dataset


def test_evaluation_dataset_is_nonempty_and_has_unique_ids() -> None:
    cases = load_dataset()

    assert len(cases) == 40
    assert len({case.case_id for case in cases}) == len(cases)
    assert {case.intent for case in cases} == {"refund", "track_order", "order", "coupon", "account"}
    assert len({case.expected_tool for case in cases}) == 30


def test_routing_baseline_is_reproducible_and_exposes_bad_cases() -> None:
    first = evaluate_dataset()
    second = evaluate_dataset()

    assert first.as_dict() == second.as_dict()
    assert first.total == 40
    assert first.top1_accuracy == 0.70
    assert first.recall_at_3 == 0.925
    assert first.recall_at_10 == 1.0
    assert {case.case_id for case in first.bad_cases} == {"cross-03", "cross-04", "cross-06"}


def test_route_policy_baseline_covers_safety_branches() -> None:
    report = evaluate_route_policy()

    assert report["accuracy"] == 1.0
    assert all(case["passed"] for case in report["cases"])
