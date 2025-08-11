#!/bin/bash
#
# Batch submission script for processing GSE datasets with intelligent recovery
# Reads gse_list.txt and submits jobs using run_quantify.sh
# 
# With the new recovery system:
# - Jobs are submitted even if partial results exist
# - The pipeline will automatically detect and recover missing samples
# - Jobs will exit cleanly if datasets are already complete
# - Only skips if mapping files are missing (run sample_mapper.py first)
#
# Usage: ./batch_submit.sh [--dry-run]
#

# Default settings
DRY_RUN=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSE_LIST="$SCRIPT_DIR/gse_list.txt"
SCRIPT="run_quantify.sh"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --file)
            GSE_LIST="$2"
            shift 2
            ;;
        --script)
            SCRIPT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run]"
            echo "Submits qsub jobs for each GSE in gse_list.txt using run_quantify.sh"
            echo ""
            echo "With intelligent recovery system:"
            echo "  - Jobs submitted even if partial results exist"
            echo "  - Pipeline automatically detects and recovers missing samples"
            echo "  - Jobs exit cleanly if datasets are already complete"
            echo "  - Only skips if mapping files are missing"
            echo ""
            echo "Options:"
            echo "  --dry-run    Show what would be submitted without actually submitting"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run]"
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ ! -f "$GSE_LIST" ]]; then
    echo "Error: GSE list file not found: $GSE_LIST"
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/$SCRIPT" ]]; then
    echo "Error: $SCRIPT not found in $SCRIPT_DIR"
    exit 1
fi

# Main execution
echo "Starting batch submission..."
echo "Reading GSE list from: $GSE_LIST"
echo "Dry run: $DRY_RUN"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Read GSE list and submit jobs
submitted_count=0
skipped_count=0

while IFS= read -r line; do
    # Skip empty lines and comments
    if [[ -z "$line" || "$line" =~ ^#.*$ ]]; then
        continue
    fi
    
    # Extract GSE ID (should be the only content on the line)
    gse=$(echo "$line" | tr -d '[:space:]')
    
    if [[ -z "$gse" ]]; then
        continue
    fi
    
    # Check if processing is needed (changed to work with recovery system)
    counts_file="data/results/${gse}/${gse}_counts.csv"
    mapping_file="data/accessions/${gse}_mapping.json"
    
    # Skip only if mapping file doesn't exist (can't process without it)
    if [[ ! -f "$mapping_file" ]]; then
        echo "Skipping $gse (no mapping file found: $mapping_file)"
        echo "  Run: python sample_mapper.py $gse"
        skipped_count=$((skipped_count + 1))
        continue
    fi
    
    # Always submit if no results file exists
    if [[ ! -f "$counts_file" ]]; then
        submit_reason="no results file"
    else
        # Results file exists - let the pipeline determine if recovery is needed
        # The new recovery system will check for missing samples and exit cleanly if complete
        submit_reason="results exist, will check for missing samples"
    fi
    
    # Submit job using the specified script
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would submit: qsub $SCRIPT $gse ($submit_reason)"
    else
        echo "Submitting job for $gse ($submit_reason)..."
        qsub "$SCRIPT" "$gse"
    fi
    
    submitted_count=$((submitted_count + 1))
    
done < "$GSE_LIST"

echo ""
echo "Batch submission complete!"
echo "Submitted: $submitted_count jobs"
echo "Skipped: $skipped_count jobs (missing mapping files)"
echo ""
echo "Note: Jobs may exit quickly if datasets are already complete."
echo "      The recovery system will automatically handle partial datasets."

if [[ "$DRY_RUN" == "false" ]]; then
    echo ""
    echo "To monitor jobs: qstat -u \$(whoami)"
    echo "Check logs in: logs/joblog_GSE*_*.log"
fi
