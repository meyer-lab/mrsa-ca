SHELL := /bin/bash

flist = $(wildcard mrsa_ca_rna/figures/figure*.py)

.PHONY: clean test all

all: $(patsubst mrsa_ca_rna/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: mrsa_ca_rna/figures/figure%.py
	@ mkdir -p ./output
	poetry run fbuild $*

clean:
	rm -rf output
