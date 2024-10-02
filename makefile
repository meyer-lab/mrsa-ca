.PHONY: clean test pyright

flist = $(wildcard mrsa_ca_rna/figures/figure*.py)

.PHONY: clean test all

all: $(patsubst mrsa_ca_rna/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: mrsa_ca_rna/figures/figure%.py
	@ mkdir -p ./output
	rye run fbuild $*

test: .venv
	rye run pytest -s -v -x

.venv: pyproject.toml
	rye sync

coverage.xml: .venv
	rye run pytest --junitxml=junit.xml --cov=mrsa_ca_rna --cov-report xml:coverage.xml

pyright: .venv
	rye run pyright mrsa_ca_rna

clean:
	rm -rf output