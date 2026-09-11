.PHONY: install test demo smoke figures clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest

demo:
	union-selection-demo --outdir outputs/demo

smoke:
	python scripts/run_from_config.py configs/smoke.json

figures:
	python scripts/make_paper_figures.py --outdir paper_figures

clean:
	rm -rf outputs paper_figures .pytest_cache build dist *.egg-info
