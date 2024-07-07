SHELL := /bin/bash

flist = $(wildcard mrsa_ca_rna/figures/figure*.py)

.PHONY: clean test all

all: $(patsubst mrsa_ca_rna/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: mrsa_ca_rna/figures/figure%.py
	@ mkdir -p ./output
	poetry run fbuild $*

clean:
	rm -rf output

mypy:
	poetry run mypy --install-types --non-interactive --ignore-missing-imports --check-untyped-defs mrsa_ca_rna
