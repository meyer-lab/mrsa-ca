#!/bin/bash

# Setup script for RNA-seq processing pipeline on Hoffman2
# This script should be run once to set up the environment

echo "Setting up RNA-seq processing pipeline environment..."

# Get current working directory for relative paths
PIPELINE_DIR="$(pwd)"
echo "Pipeline directory: $PIPELINE_DIR"

#!/bin/bash

# Setup script for RNA-seq processing pipeline on Hoffman2
# This script should be run once to set up the environment

echo "Setting up RNA-seq processing pipeline environment..."

# Get current working directory for relative paths
PIPELINE_DIR="$(pwd)"
echo "Pipeline directory: $PIPELINE_DIR"

# Create necessary directories
mkdir -p logs
mkdir -p data/accessions
mkdir -p data/results

# Create user-specific scratch and output directories using Hoffman2 environment variables
# $SCRATCH and $HOME are provided by the cluster
echo "Using Hoffman2 environment variables:"
echo "  SCRATCH: $SCRATCH"
echo "  HOME: $HOME"

# Create necessary directories using environment variables
mkdir -p "${SCRATCH}/rnaseq_fastq"
# Results are now stored directly in the pipeline directory
mkdir -p "${PIPELINE_DIR}/data/results"

echo "Created Hoffman2 directories:"
echo "  Scratch (FASTQ): ${SCRATCH}/rnaseq_fastq"
echo "  Results: ${PIPELINE_DIR}/data/results"

# Set environment variables for the pipeline
export RNASEQ_SCRATCH_DIR="${SCRATCH}/rnaseq_fastq"
# Results now go directly to pipeline directory
export RNASEQ_RESULTS_DIR="${PIPELINE_DIR}/data/results"

# Check if Python virtual environment exists in a few common locations
VENV_PATHS=(
    "${HOME}/.venv"
    "${HOME}/venv"
    "${PIPELINE_DIR}/.venv"
    "${PIPELINE_DIR}/venv"
)

VENV_FOUND=false
for venv_path in "${VENV_PATHS[@]}"; do
    if [ -d "$venv_path" ]; then
        echo "Found virtual environment at: $venv_path"
        source "$venv_path/bin/activate"
        VENV_FOUND=true
        break
    fi
done

if [ "$VENV_FOUND" = false ]; then
    echo "No virtual environment found. Checking if Python modules are available..."
    echo ""
    echo "If you need to create a virtual environment:"
    echo "  python -m venv ~/.venv"
    echo "  source ~/.venv/bin/activate"
    echo "  pip install -r requirements.txt"
    echo ""
    echo "Continuing with system Python..."
fi

echo "Checking Python dependencies..."
python -c "import pandas; print(f'✅ pandas version: {pandas.__version__}')" || echo "❌ pandas not installed"
python -c "import requests; print('✅ requests available')" || echo "❌ requests not installed"
python -c "import certifi; print('✅ certifi available')" || echo "❌ certifi not installed"
python -c "import xalign; print(f'✅ xalign available')" || echo "❌ xalign not installed"

# Install SRA Toolkit
echo ""
echo "Installing SRA Toolkit..."
if [ -f "utils/sra_toolkit_installer.py" ]; then
    python utils/sra_toolkit_installer.py --install
    if [ $? -eq 0 ]; then
        echo "✅ SRA Toolkit installed successfully"
    else
        echo "⚠️  SRA Toolkit installation failed"
        echo "   You may need to install it manually or check network connectivity"
    fi
else
    echo "⚠️  SRA Toolkit installer not found"
    echo "   Manual installation may be required"
fi

# Check XAlign index status
echo ""
echo "Checking XAlign reference index status..."
if [ -f "utils/xalign_index_validator.py" ]; then
    python utils/xalign_index_validator.py --report
else
    echo "ℹ️  XAlign validator not found. Indexes will be built automatically on first use."
fi

# Check if the script is executable
if [ ! -x "process_rnaseq.py" ]; then
    chmod +x process_rnaseq.py
    echo "Made process_rnaseq.py executable"
fi

if [ ! -x "run_quantify.sh" ]; then
    chmod +x run_quantify.sh
    echo "Made run_quantify.sh executable"
fi

echo "Setup completed!"
echo ""
echo "Environment variables set:"
echo "  RNASEQ_SCRATCH_DIR: $RNASEQ_SCRATCH_DIR"
echo "  RNASEQ_RESULTS_DIR: $RNASEQ_RESULTS_DIR"
echo ""
echo "Directory structure created:"
echo "  📁 ${PIPELINE_DIR}/logs/ - Job and processing logs"
echo "  📁 ${PIPELINE_DIR}/data/accessions/ - Study mapping files"
echo "  📁 ${PIPELINE_DIR}/data/results/ - Final compiled results (stored directly)"
echo "  📁 ${SCRATCH}/rnaseq_fastq/ - Temporary FASTQ files (per study)"
echo ""
echo "To run the pipeline:"
echo "1. Prepare your sample mapping file (JSON format)"
echo "2. Customize run_quantify.sh with your email and study details"
echo "3. Submit the job: qsub run_quantify.sh STUDY_ID mapping_file.json"
echo "4. Results will be available directly in: ${PIPELINE_DIR}/data/results/STUDY_ID/"
echo ""
echo "For detailed usage examples, see: docs/USAGE_EXAMPLES.md"
