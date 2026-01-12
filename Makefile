.PHONY: help train-detection eval-detection lint test clean

PYTHONPATH=.

help:
	@echo "Available commands:"
	@echo "  make train-detection   Train detection model"
	@echo "  make eval-detection    Run detection evaluation"
	@echo "  make lint              Run basic lint checks"
	@echo "  make test              Run unit tests"
	@echo "  make clean             Remove artifacts"

train-detection:
	PYTHONPATH=$(PYTHONPATH) python src/training/train_detection_full.py

eval-detection:
	PYTHONPATH=$(PYTHONPATH) python src/training/test_evaluation.py

lint:
	python -m compileall src

test:
	PYTHONPATH=$(PYTHONPATH) python src/training/test_evaluation.py

clean:
	rm -rf outputs/models/*.pt

