import json
from pathlib import Path


def test_confirmatory_freeze_has_paper_design() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/paper_confirmatory.json").read_text())
    expected = json.loads((root / "paper_reference/expected_confirmatory_summary.json").read_text())
    assert config["profile"] == "confirmatory"
    assert config["bins_values"] == [12, 16, 20]
    assert config["primary_bins"] == 16
    assert config["seed"] == 20260912
    assert expected["n_successful_full_fits"] == 1080
    assert expected["n_successful_cv_fits"] == 8100
    assert expected["n_failed_fits"] == 0
