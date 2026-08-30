# CROSSING — common workflows
.PHONY: install test train demo serve evaluate clean fmt

install:
	python3 -m pip install -r requirements.txt
	python3 -m pip install -e .

test:
	python3 -m pytest -q

# fast subset (no heavy pipeline runs)
test-fast:
	python3 -m pytest tests/test_geometry.py tests/test_nn_gradcheck.py -q

train:
	python3 -m scripts.train_all --duration 30 --train 8

train-fast:
	python3 -m scripts.train_all --fast

demo:
	python3 -m scripts.build_demo --seed 7 --duration 45

serve:
	python3 -m uvicorn crossing.api.server:app --host 127.0.0.1 --port 8000

evaluate:
	python3 -m crossing.cli evaluate --seeds 201 202 203

clean:
	rm -rf artifacts/models/*.pkl artifacts/models/*.npz artifacts/metrics/*.json artifacts/db/*.db
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
