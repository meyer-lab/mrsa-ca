.PHONY: all clean_output test pyright setup_salmon quantify_salmon clean_salmon help

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
	PYTHONPATH=. rye run python -c "from mrsa_ca_rna.experiments.wandb_$* import perform_experiment; perform_experiment()" > ./sweep_logs/$*_$(shell date +%Y%m%d%_H%M%S).log 2>&1

.venv: pyproject.toml
	rye sync

coverage.xml: .venv
	rye run pytest --junitxml=junit.xml --cov=mrsa_ca_rna --cov-report xml:coverage.xml

pyright: .venv
	rye run pyright mrsa_ca_rna

clean_output:
	rm -rf output

# Makefile for RNA-Seq analysis pipeline using Salmon
# Default number of threads to use
THREADS ?= 4
# Default to paired-end reads unless specified otherwise
ENDTYPE ?= paired
# Default batch size for processing
BATCH_SIZE ?= 5
# Default project name
PROJECT ?= $(shell date +%Y%m%d)

# Directories
SALMON_DIR := salmon_processing
SRA_DIR := $(SALMON_DIR)/sra_out
REF_DIR := $(SALMON_DIR)/salmon_ref
COUNTS_DIR := $(SALMON_DIR)/salmon_gene_counts

# Reference files
TRANSCRIPTS := $(REF_DIR)/human_transcripts.fa.gz
GENOME := $(REF_DIR)/human_genome.fa.gz
GENTROME := $(REF_DIR)/human_gentrome.fa.gz
DECOYS := $(REF_DIR)/decoys.txt
SALMON_INDEX := $(REF_DIR)/salmon_index
GTF := $(REF_DIR)/mappings.gtf

# Setup directories and reference files
setup_salmon: $(SALMON_DIR) $(SRA_DIR) $(REF_DIR) $(COUNTS_DIR) $(SALMON_INDEX) $(GTF)
	@echo "Setup complete. Directory structure created and reference files downloaded."
	@echo "Next steps:"
	@echo "1. Create your accession list file at: $(SRA_DIR)/accession_list.txt"
	@echo "2. Run quantification with: make quantify THREADS=<num> ENDTYPE=<single|paired> BATCH_SIZE=<num>"

# Create directories
$(SALMON_DIR):
	mkdir -p $@

$(SRA_DIR): | $(SALMON_DIR)
	mkdir -p $@

$(REF_DIR): | $(SALMON_DIR)
	mkdir -p $@

$(COUNTS_DIR): | $(SALMON_DIR)
	mkdir -p $@

# Download and prepare reference files
$(TRANSCRIPTS): | $(REF_DIR)
	curl https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.transcripts.fa.gz -o $@

$(GENOME): | $(REF_DIR)
	curl https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/GRCh38.p14.genome.fa.gz -o $@

$(GTF).gz: | $(REF_DIR)
	curl https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.annotation.gtf.gz -o $@

$(GTF): $(GTF).gz
	gzip -d $<

$(DECOYS): $(GENOME) | $(REF_DIR)
	gunzip -c $(GENOME) | grep "^>" | cut -d " " -f 1 | sed 's/>//g' > $@

$(GENTROME): $(TRANSCRIPTS) $(GENOME) | $(REF_DIR)
	cat $(TRANSCRIPTS) $(GENOME) > $@

$(SALMON_INDEX): $(GENTROME) $(DECOYS)
	salmon index -t $(GENTROME) -d $(DECOYS) --gencode -p $(THREADS) -i $@

# Quantification rules
quantify_salmon: $(COUNTS_DIR)/all_counts_$(PROJECT).txt

# Process SRA accessions and run Salmon quantification
$(COUNTS_DIR)/all_counts_$(PROJECT).txt: $(SRA_DIR)/accession_list.txt | $(COUNTS_DIR)
	@echo "Processing SRA accessions in batches of $(BATCH_SIZE)..."
	@# Calculate total number of accessions and batches
	@total_lines=$$(wc -l < $(SRA_DIR)/accession_list.txt); \
	batch_count=$$(( (total_lines + $(BATCH_SIZE) - 1) / $(BATCH_SIZE) )); \
	for batch_num in $$(seq 1 $$batch_count); do \
		start_line=$$(( (batch_num - 1) * $(BATCH_SIZE) + 1 )); \
		end_line=$$(( batch_num * $(BATCH_SIZE) )); \
		if [ $$end_line -gt $$total_lines ]; then \
			end_line=$$total_lines; \
		fi; \
		echo "Processing batch $$batch_num of $$batch_count (lines $$start_line to $$end_line)"; \
		sed -n "$$start_line,$$end_line p" $(SRA_DIR)/accession_list.txt > $(SRA_DIR)/batch_$$batch_num.txt; \
		while read -r acc; do \
			if [ "$(ENDTYPE)" = "single" ]; then \
				fasterq-dump $$acc --progress --threads $(THREADS) -O $(SRA_DIR) -t $(SRA_DIR)/temp & \
			else \
				fasterq-dump $$acc --split-files --skip-technical --progress --threads $(THREADS) -O $(SRA_DIR) -t $(SRA_DIR)/temp & \
			fi; \
		done < $(SRA_DIR)/batch_$$batch_num.txt; \
		wait; \
		echo "Quantifying batch $$batch_num..."; \
		while read -r acc; do \
			if [ "$(ENDTYPE)" = "single" ]; then \
				salmon quant -p $(THREADS) -i $(SALMON_INDEX) \
					--geneMap $(GTF) --validateMappings --gcBias -l A \
					-r $(SRA_DIR)/$$acc.fastq \
					-o $(COUNTS_DIR)/$$acc; \
			else \
				salmon quant -p $(THREADS) -i $(SALMON_INDEX) \
					--geneMap $(GTF) --validateMappings --gcBias -l A \
					-1 $(SRA_DIR)/$${acc}_1.fastq -2 $(SRA_DIR)/$${acc}_2.fastq \
					-o $(COUNTS_DIR)/$$acc; \
			fi; \
		done < $(SRA_DIR)/batch_$$batch_num.txt; \
		rm $(SRA_DIR)/*.fastq $(SRA_DIR)/batch_$$batch_num.txt; \
	done
	@echo "Aggregating gene counts..."
	@head -n 1 $(SRA_DIR)/accession_list.txt | while read -r sample; do \
		tail -n +2 $(COUNTS_DIR)/$$sample/quant.genes.sf | cut -f 1 > $(COUNTS_DIR)/genes.txt; \
	done
	@while read -r sample; do \
		tail -n +2 $(COUNTS_DIR)/$$sample/quant.genes.sf | cut -f 4 > $(COUNTS_DIR)/$$sample.count; \
	done < $(SRA_DIR)/accession_list.txt
	@paste $(COUNTS_DIR)/genes.txt $(COUNTS_DIR)/*.count > $@
	@sed -i "1i gene\t$$(sort $(SRA_DIR)/accession_list.txt | tr '\n' '\t')" $@
	@rm -r $(COUNTS_DIR)/!(all_counts_$(PROJECT).txt)

# Clean up
clean_salmon:
	rm -rf $(SALMON_DIR)

# Help target
help:
	@echo ""
	@echo "MRSA-CA project Makefile"
	@echo " make help		- Show this help message"
	@echo " make test		- Run tests"
	@echo " make pyright		- Run Pyright static type checker"
	@echo " make all		- Run all figure scripts"
	@echo " make ./output/figure%.svg	- Run a specific figure script"
	@echo " make clean_output		- Remove all generated files"
	@echo ""
	@echo ""
	@echo "RNA-Seq Analysis Pipeline Makefile"
	@echo "Usage:"
	@echo "  make setup_salmon THREADS=<num_threads>		- Set up directories and download reference files"
	@echo "  make quantify_salmon ENDTYPE=<single|paired> THREADS=<num_threads> BATCH_SIZE=<num> PROJECT=<your_project>		- Run Salmon quantification"
	@echo "  make clean_salmon		- Remove all generated files"
	@echo "  make Help		- Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  THREADS			- Number of threads to use (default: 4)"
	@echo "  ENDTYPE			- Type of sequencing reads: 'single' or 'paired' (default: paired)"
	@echo "  BATCH_SIZE			- Number of accessions to process simultaneously (default: 5)"
	@echo "  PROJECT			- Name of the project (default: current date)"
	@echo ""