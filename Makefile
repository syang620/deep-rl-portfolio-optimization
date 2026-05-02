.PHONY: etl features test validate phase1-check phase1

PYTHON := .venv/bin/python

etl:
	$(PYTHON) scripts/run_etl.py --config configs/data.yaml --universe configs/universe.yaml

features:
	$(PYTHON) scripts/build_features.py --data-config configs/data.yaml --feature-config configs/features.yaml --universe configs/universe.yaml

test:
	$(PYTHON) -m pytest

validate:
	$(PYTHON) scripts/validate_phase1.py

phase1-check: validate

phase1: etl features validate test
