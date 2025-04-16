.PHONY: all clean_output test pyright

flist = $(wildcard mrsa_ca_rna/figures/figure*.py)

all: $(patsubst mrsa_ca_rna/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: mrsa_ca_rna/figures/figure%.py
	@ mkdir -p ./output
	rye run fbuild $*

test: .venv
	rye run pytest -s -v -x

wandb_%: .venv
	@echo "Running wandb experiment: $*"
	mkdir -p ./wandb_logs
	PYTHONPATH=. rye run python -c "from mrsa_ca_rna.experiments.wandb_$* import perform_experiment; perform_experiment()" > ./wandb_logs/$*_$(shell date +%Y%m%d%_H%M%S).log 2>&1

.venv: pyproject.toml
	rye sync

coverage.xml: .venv
	rye run pytest --junitxml=junit.xml --cov=mrsa_ca_rna --cov-report xml:coverage.xml

pyright: .venv
	rye run pyright mrsa_ca_rna

format: .venv
	rye fmt
	rye lint --fix

clean_output:
	rm -rf output
