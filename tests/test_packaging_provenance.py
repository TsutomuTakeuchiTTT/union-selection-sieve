from pathlib import Path


def test_import_safe_core_differs_only_by_header_and_autorun_guard() -> None:
    root = Path(__file__).resolve().parents[1]
    archival = (root / "reference/confirmatory_v1_1/union_selection_sieve_confirmatory_monte_carlo_v1_1_onecell.py").read_text()
    packaged = (root / "src/union_selection_sieve/_reference_impl.py").read_text()
    header = (
        "# NOTE: This module is derived verbatim from the validated one-cell\n"
        "# confirmatory implementation, except that its final autorun block is guarded\n"
        "# by ``if __name__ == \"__main__\"`` so it can be imported safely as a package.\n"
        "# The unmodified archival source is stored under reference/confirmatory_v1_1/.\n\n"
    )
    assert packaged.startswith(header)
    restored = packaged[len(header):].replace(
        'if __name__ == "__main__" and os.environ.get("UNION_SELECTION_CONFIRMATORY_SKIP_AUTORUN", "0") != "1":',
        'if os.environ.get("UNION_SELECTION_CONFIRMATORY_SKIP_AUTORUN", "0") != "1":',
        1,
    )
    assert restored == archival
