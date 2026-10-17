PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
CMAKE ?= cmake

.PHONY: test test-python test-cpp demo

# Build moe_run first. This demo needs no Torch, model download, GPU or network.
demo:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 $(PYTHON) -m evaluation.demo

test: test-python

test-python: test-cpp
	$(PYTHON) -m pytest -q

test-cpp:
	$(CMAKE) -S port -B port/build
	$(CMAKE) --build port/build
	$(CMAKE) --build port/build --target test
