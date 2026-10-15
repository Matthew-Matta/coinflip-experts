PYTHON ?= python3
CMAKE ?= cmake

.PHONY: test test-python test-cpp

test: test-python test-cpp

test-python:
	$(PYTHON) -m pytest -q

test-cpp:
	$(CMAKE) -S port -B port/build
	$(CMAKE) --build port/build
	$(CMAKE) --build port/build --target test
