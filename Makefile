.PHONY: setup data train test serve dev-ui docker
PY ?= python3

setup:            ## install runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt

data:             ## fetch Leffingwell via Pyrfume and build the processed dataset + ledger
	PYTHONPATH=src:. $(PY) -m ml.data.build_dataset --fetch

train:            ## train, evaluate and write a release bundle to artifacts/releases/<version>
	PYTHONPATH=src:. $(PY) -m ml.train.train --version 0.1.0

test:
	PYTHONPATH=src:. $(PY) -m pytest -q

serve:            ## API + web client on http://localhost:8000
	PYTHONPATH=src:. AROMASCOPE_SERVE_WEB=apps/web AROMASCOPE_ORIGINS="*" $(PY) -m uvicorn services.api.app.main:app --port 8000

dev-ui:           ## work on the web client without RDKit: mock API + client on http://localhost:8080
	$(PY) tools/mock_api.py

docker:
	docker build -t aromascope . && docker run --rm -p 7860:7860 aromascope
