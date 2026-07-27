PDF ?= docs/SEBU7844-37.pdf

.PHONY: ingest verify clean install

install:
	pip install -r requirements.txt

ingest:
	python3 -m ingestion.cli $(if $(F),$(F),$(PDF))

verify:
	python3 verify_chunks.py $(if $(F),$(F),$(PDF))

clean:
	rm -f chunks.json
	find . -name __pycache__ -type d -exec rm -rf {} +
