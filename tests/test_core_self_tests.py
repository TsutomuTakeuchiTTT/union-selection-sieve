from union_selection_sieve import run_self_tests


def test_validated_core_self_tests_pass() -> None:
    result = run_self_tests()
    assert result["all_passed"]
    assert all(result["checks"].values())
