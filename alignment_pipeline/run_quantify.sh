#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o logs/
#$ -j y
## Edit the line below as needed:
#$ -l h_rt=23:59:59,h_data=2G
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 16
# Email address to notify
#$ -M jrpopoli@g.ucla.edu
# Notify when
#$ -m bea

# Check for required study ID parameter
if [ -z "$1" ]; then
    echo "ERROR: Study ID required!"
    echo "Usage: qsub run_quantify.sh <STUDY_ID> [mapping_file_or_srr_file]"
    echo "Examples:"
    echo "  qsub run_quantify.sh GSE12345 my_study_mapping.json  # Sample mapping mode"
    echo "  qsub run_quantify.sh MyStudy my_srr_list.txt         # Direct SRR mode"
    echo "  qsub run_quantify.sh GSE12345                        # Auto-find GSE mapping"
    exit 1
fi

STUDY_ID="$1"
INPUT_FILE=""
PROCESSING_MODE=""

# Redirect all output to a properly named log file
LOG_FILE="logs/joblog_${STUDY_ID}_${JOB_ID}.log"
exec 1> "$LOG_FILE" 2>&1

# Determine processing mode based on second parameter or auto-detection
if [ -z "$2" ]; then
    # No second parameter - look for GSE mapping file
    MAPPING_FILE="${STUDY_ID}_mapping.json"
    # First try current directory
    if [ ! -f "$MAPPING_FILE" ]; then
        # Then try data/accessions/
        MAPPING_FILE="data/accessions/${STUDY_ID}_mapping.json"
    fi
    
    if [ -f "$MAPPING_FILE" ]; then
        INPUT_FILE="$MAPPING_FILE"
        PROCESSING_MODE="sample_mapping"
    else
        echo "ERROR: No mapping file found for $STUDY_ID"
        echo "Checked locations:"
        echo "  - Current directory: $(pwd)/$MAPPING_FILE"
        echo "  - Accessions directory: data/accessions/${STUDY_ID}_mapping.json"
        exit 1
    fi
else
    INPUT_FILE="$2"
    
    # Detect mode based on file extension and content
    if [[ "$INPUT_FILE" == *.json ]] || [[ "$INPUT_FILE" == *.csv ]] || [[ "$INPUT_FILE" == *.tsv ]]; then
        PROCESSING_MODE="sample_mapping"
    else
        # Assume it's an SRR list file
        PROCESSING_MODE="srr_file"
    fi
fi

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo "Processing Study: $STUDY_ID"
echo "Processing mode: $PROCESSING_MODE"
echo "Using input file: $INPUT_FILE"
echo " "

# load the job environment:
. /u/local/Modules/default/init/modules.sh
## Edit the line below as needed:
module load gcc/11.3.0
module load cmake
module load python

# Add SRA toolkit to PATH
export PATH="/u/home/j/jrpopoli/sratoolkit.2.10.8-ubuntu64/bin:$PATH"
echo "Added SRA toolkit to PATH"

# Verify fasterq-dump is available
if command -v fasterq-dump &> /dev/null; then
    echo "✅ fasterq-dump available"
else
    echo "❌ fasterq-dump not found in PATH"
fi

# Set up optimized threading for 8-core job allocation
echo "Configuring thread allocation for performance optimization..."

# Get the number of allocated cores from SGE
NSLOTS=${NSLOTS:-8}  # Default to 8 if not set by SGE
echo "SGE allocated cores: $NSLOTS"

# Intelligent core distribution:
# - Reserve 1 core for pipeline orchestration
# - Allocate 2-3 cores for parallel SRA downloads
# - Allocate remaining cores for kallisto alignment threading
# - Conservative threading for mathematical libraries

AVAILABLE_CORES=$((NSLOTS - 1))  # Reserve 1 for orchestration
DOWNLOAD_THREADS=$(( AVAILABLE_CORES < 3 ? 1 : 3 ))  # Min 1, max 3
ALIGNMENT_THREADS=$((AVAILABLE_CORES - DOWNLOAD_THREADS + 1))  # Rest to alignment
MATH_THREADS=$((NSLOTS / 4))  # Conservative for NumPy/pandas
MATH_THREADS=$(( MATH_THREADS < 1 ? 1 : MATH_THREADS ))  # At least 1

# Export thread configuration for the pipeline
export PIPELINE_DOWNLOAD_THREADS=$DOWNLOAD_THREADS
export PIPELINE_ALIGNMENT_THREADS=$ALIGNMENT_THREADS
export OMP_NUM_THREADS=$MATH_THREADS
export OPENBLAS_NUM_THREADS=$MATH_THREADS
export MKL_NUM_THREADS=$MATH_THREADS

echo "Thread allocation configured:"
echo "  - Download threads: $DOWNLOAD_THREADS"
echo "  - Alignment threads: $ALIGNMENT_THREADS" 
echo "  - Math library threads: $MATH_THREADS"
echo "  - Total cores utilized: $NSLOTS"

# Use Hoffman2 environment variables for directory paths
# $SCRATCH and $HOME are provided by the cluster environment
SCRATCH_DIR="${SCRATCH}/rnaseq_fastq/${STUDY_ID}"

# Capture current working directory as pipeline base
PIPELINE_DIR="$(pwd)"

# Store results directly in the pipeline data directory
RESULTS_DIR="${PIPELINE_DIR}/data/results/${STUDY_ID}"

# Create necessary directories
mkdir -p logs
mkdir -p "${SCRATCH_DIR}"
mkdir -p "${RESULTS_DIR}"

# Check disk space usage against quotas
echo "Checking disk usage against quotas..."
echo ""

# Check SCRATCH usage (2TB quota)
SCRATCH_USAGE=$(du -sh "${SCRATCH}" 2>/dev/null | cut -f1)
echo "SCRATCH usage: ${SCRATCH_USAGE} / 2TB quota"
echo "SCRATCH location: ${SCRATCH}"

# Check HOME/Workspaces usage (40GB quota)
WORKSPACES_DIR="${HOME}/Workspaces"
if [ -d "${WORKSPACES_DIR}" ]; then
    WORKSPACES_USAGE=$(du -sh "${WORKSPACES_DIR}" 2>/dev/null | cut -f1)
    echo "Workspaces usage: ${WORKSPACES_USAGE} / 40GB quota"
    echo "Workspaces location: ${WORKSPACES_DIR}"
else
    echo "Workspaces directory not found: ${WORKSPACES_DIR}"
fi

echo ""

# Activate virtual environment (assume it exists as per requirements)
# Check common locations for virtual environment
VENV_PATHS=(
    "${HOME}/.venv"
    "${HOME}/venv"
    "${PIPELINE_DIR}/.venv"
    "${PIPELINE_DIR}/venv"
)

# Find and activate virtual environment
for venv_path in "${VENV_PATHS[@]}"; do
    if [ -d "$venv_path" ]; then
        echo "Activating virtual environment at: $venv_path"
        source "$venv_path/bin/activate"
        break
    fi
done

# Verify xalign is available (critical package for pipeline)
python -c "import xalign; print('✅ xalign available')" || {
    echo "❌ ERROR: xalign not available in current environment"
    echo "Please ensure virtual environment with xalign is properly set up"
    exit 1
}

# Look for input file in current directory, then in data/accessions/
if [ ! -f "$INPUT_FILE" ]; then
    # Try data/accessions/ directory if not found in current directory
    if [[ "$PROCESSING_MODE" == "sample_mapping" ]]; then
        ACCESSIONS_FILE="${PIPELINE_DIR}/data/accessions/$(basename $INPUT_FILE)"
        if [ -f "$ACCESSIONS_FILE" ]; then
            INPUT_FILE="$ACCESSIONS_FILE"
            echo "Found input file in accessions directory: $INPUT_FILE"
        else
            echo "ERROR: Input file $INPUT_FILE not found!"
            echo "Checked locations:"
            echo "  - Current directory: $(pwd)/$INPUT_FILE"
            echo "  - Accessions directory: $ACCESSIONS_FILE"
            echo "Available mapping files in data/accessions/:"
            ls -la "${PIPELINE_DIR}/data/accessions/"*mapping*.json 2>/dev/null || echo "No mapping files found"
            exit 1
        fi
    else
        # Try data/accessions/ directory for SRR files too
        ACCESSIONS_FILE="${PIPELINE_DIR}/data/accessions/$(basename $INPUT_FILE)"
        if [ -f "$ACCESSIONS_FILE" ]; then
            INPUT_FILE="$ACCESSIONS_FILE"
            echo "Found SRR file in accessions directory: $INPUT_FILE"
        else
            echo "ERROR: SRR file $INPUT_FILE not found!"
            echo "Checked locations:"
            echo "  - Current directory: $(pwd)/$INPUT_FILE"
            echo "  - Accessions directory: $ACCESSIONS_FILE"
            echo "Available SRR files in data/accessions/:"
            ls -la "${PIPELINE_DIR}/data/accessions/"*.txt 2>/dev/null || echo "No .txt files found"
            exit 1
        fi
    fi
fi

# Display input file content for verification
if [[ "$PROCESSING_MODE" == "sample_mapping" ]]; then
    echo "Processing the following sample mapping:"
    cat "$INPUT_FILE"
else
    echo "Processing the following SRR accessions:"
    head -10 "$INPUT_FILE"
    if [ $(wc -l < "$INPUT_FILE") -gt 10 ]; then
        echo "... and $(( $(wc -l < "$INPUT_FILE") - 10 )) more SRRs"
    fi
fi

# Run the processing script with appropriate parameters based on mode
if [[ "$PROCESSING_MODE" == "sample_mapping" ]]; then
    echo "Running in sample mapping mode with simplified CSV-based checkpointing..."
    python process_rnaseq.py \
        --sample_mapping "$INPUT_FILE" \
        --input_dir "$SCRATCH_DIR" \
        --output_dir "$RESULTS_DIR" \
        --study_id "$STUDY_ID" \
        --max_workers 1 \
        --genome homo_sapiens \
        --return_type gene \
        --identifier symbol \
        --cleanup immediate \
        --combination_method sum
else
    echo "Running in direct SRR mode with simplified CSV-based checkpointing..."
    python process_rnaseq.py \
        --srr_file "$INPUT_FILE" \
        --input_dir "$SCRATCH_DIR" \
        --output_dir "$RESULTS_DIR" \
        --study_id "$STUDY_ID" \
        --max_workers 1 \
        --genome homo_sapiens \
        --return_type gene \
        --identifier symbol \
        --cleanup immediate
fi

# echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
echo " "

# Clean up scratch space if successful
if [ $? -eq 0 ]; then
    echo "Processing completed successfully. Cleaning up scratch space..."
    # Clean up FASTQ files from scratch to save space (results are in permanent storage)
    rm -rf "${SCRATCH_DIR}"/*.fastq
    
    echo ""
    echo "✅ Processing completed successfully!"
    echo "📁 Results available in: $RESULTS_DIR"
    echo ""
    echo "Key files in results directory:"
    ls -la "$RESULTS_DIR"
    
else
    echo "Processing failed. Check the logs for errors."
    echo "FASTQ files preserved in scratch for debugging: $SCRATCH_DIR"
    echo "Partial results may be in: $RESULTS_DIR"
    echo "Intermediate CSV files preserved for recovery on restart"
fi
