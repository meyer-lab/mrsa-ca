#!/usr/bin/env python3
"""
Study Results Concatenator

This script concatenates all study results CSV files into a single master file
with the following transformation:
- Input: genes as rows, GSM samples as columns (current format)
- Output: GSM samples as rows, genes as columns (desired format)

This makes it easy to:
1. Transfer all results at once
2. Use GSM numbers as index for matching with metadata
3. Have genes as columns for downstream analysis

Usage:
    python concatenate_studies.py [--input-dir data/results] \
        [--output master_expression_data.csv] [--exclude-dirs examine,example]
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_study_files(input_dir: str, exclude_dirs: list[str] = None) -> list[Path]:
    """
    Find all *_counts.csv files in the results directory.

    Args:
        input_dir: Base directory containing study results
        exclude_dirs: List of directory names to exclude
                     (e.g., 'examine', 'example')

    Returns:
        List of Path objects for found CSV files
    """
    if exclude_dirs is None:
        exclude_dirs = ["examine", "example"]

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all *_counts.csv files recursively
    csv_files = list(input_path.rglob("*_counts.csv"))

    # Filter out files in excluded directories
    filtered_files = []
    for csv_file in csv_files:
        # Check if any parent directory is in exclude_dirs
        if not any(exclude_dir in csv_file.parts for exclude_dir in exclude_dirs):
            filtered_files.append(csv_file)

    logger.info(
        f"Found {len(csv_files)} total CSV files, {len(filtered_files)} after filtering"
    )

    if exclude_dirs:
        excluded_count = len(csv_files) - len(filtered_files)
        logger.info(
            f"Excluded {excluded_count} files from directories: "
            f"{', '.join(exclude_dirs)}"
        )

    return filtered_files


def load_and_transpose_study(csv_file: Path) -> pd.DataFrame | None:
    """
    Load a study CSV file and transpose it to have GSM samples as rows.

    Args:
        csv_file: Path to the study CSV file

    Returns:
        Transposed DataFrame with GSM samples as index and genes as columns,
        or None if loading failed
    """
    try:
        # Extract study ID from filename
        study_id = csv_file.stem.replace("_counts", "")

        logger.info(f"Loading {study_id}...")

        # Load the CSV file
        df = pd.read_csv(csv_file, index_col=0)  # Use first column as index

        # Transpose so that genes become columns and samples become rows
        transposed_df = df.T

        # Set the index name to 'GSM'
        transposed_df.index.name = "GSM"

        # Add study_id as a column for tracking
        transposed_df["study_id"] = study_id

        logger.info(
            f"  Loaded {study_id}: {transposed_df.shape[0]} samples, "
            f"{transposed_df.shape[1] - 1} genes"
        )

        return transposed_df

    except Exception as e:
        logger.error(f"Failed to load {csv_file}: {e}")
        return None


def concatenate_studies(study_dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate all study DataFrames into a single master DataFrame.

    Args:
        study_dataframes: List of transposed study DataFrames

    Returns:
        Concatenated DataFrame with all samples
    """
    if not study_dataframes:
        raise ValueError("No study DataFrames provided")

    logger.info("Concatenating all studies...")

    # Concatenate all DataFrames
    master_df = pd.concat(study_dataframes, axis=0, sort=False)

    # Check for duplicate GSM indices
    duplicate_gsms = master_df.index.duplicated()
    if duplicate_gsms.any():
        duplicate_count = duplicate_gsms.sum()
        logger.warning(f"Found {duplicate_count} duplicate GSM samples")

        # Show which GSMs are duplicated
        duplicated_gsms = master_df.index[duplicate_gsms].unique()
        logger.warning(
            f"Duplicated GSMs: {list(duplicated_gsms)[:10]}"
            f"{'...' if len(duplicated_gsms) > 10 else ''}"
        )

        # Keep first occurrence of each duplicate
        master_df = master_df[~duplicate_gsms]
        logger.info("Removed duplicates, keeping first occurrence of each")

    # Move study_id column to the front
    if "study_id" in master_df.columns:
        cols = ["study_id"] + [col for col in master_df.columns if col != "study_id"]
        master_df = master_df[cols]

    logger.info(
        f"Master dataset: {master_df.shape[0]} samples, {master_df.shape[1] - 1} genes"
    )

    return master_df


def generate_summary_report(master_df: pd.DataFrame, study_files: list[Path]) -> str:
    """Generate a summary report of the concatenation process."""

    report = f"""
        {"=" * 70}
        STUDY CONCATENATION SUMMARY REPORT
        {"=" * 70}

        INPUT:
        Total study files found: {len(study_files)}
        Studies included:

    """

    # List all studies with sample counts
    if "study_id" in master_df.columns:
        study_counts = master_df["study_id"].value_counts().sort_index()
        for study_id, count in study_counts.items():
            report += f"    {study_id}: {count} samples\n"

    report += f"""
        OUTPUT:
        Total samples: {master_df.shape[0]:,}
        Total genes: {master_df.shape[1] - 1:,} (excluding study_id column)
        
        DATA QUALITY:
        Missing values: {master_df.isnull().sum().sum():,}
        Duplicate GSM samples: {
        "None" if not master_df.index.duplicated().any() else "Found (handled)"
    }
        
        SAMPLE STATISTICS:
        Min expression per sample: {master_df.select_dtypes(include=[np.number]).sum(axis=1).min():,.0f}
        Max expression per sample: {master_df.select_dtypes(include=[np.number]).sum(axis=1).max():,.0f}
        Median expression per sample: {master_df.select_dtypes(include=[np.number]).sum(axis=1).median():,.0f}

        GENE STATISTICS:
        Min expression per gene: {master_df.select_dtypes(include=[np.number]).sum(axis=0).min():,.0f}
        Max expression per gene: {master_df.select_dtypes(include=[np.number]).sum(axis=0).max():,.0f}
        Median expression per gene: {master_df.select_dtypes(include=[np.number]).sum(axis=0).median():,.0f}

        {"=" * 70}
    """

    return report


def save_master_file(
    master_df: pd.DataFrame, output_file: str, compression: str = None
):
    """
    Save the master DataFrame to file with optional compression.

    Args:
        master_df: Master DataFrame to save
        output_file: Output file path
        compression: Compression type ('gzip', 'bz2', 'xz', or None)
    """
    logger.info(f"Saving master file to: {output_file}")

    if compression:
        if not output_file.endswith(f".{compression}"):
            output_file += f".{compression}"
        logger.info(f"Using {compression} compression")

    try:
        master_df.to_csv(output_file, compression=compression)

        # Get file size
        file_size = os.path.getsize(output_file)
        file_size_mb = file_size / (1024 * 1024)

        logger.info(f"Successfully saved: {file_size_mb:.1f} MB")

    except Exception as e:
        logger.error(f"Failed to save master file: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate RNA-seq study results into a single master file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python concatenate_studies.py
            python concatenate_studies.py --output master_data.csv
            python concatenate_studies.py --exclude-dirs examine,example,test
            python concatenate_studies.py --compress gzip
        """,
    )

    parser.add_argument(
        "--input-dir",
        default="data/results",
        help="Input directory containing study results (default: data/results)",
    )

    parser.add_argument(
        "--output",
        default="master_expression_data.csv",
        help="Output file name (default: master_expression_data.csv)",
    )

    parser.add_argument(
        "--exclude-dirs",
        default="examine,example",
        help="Comma-separated list of directory names to exclude "
        "(default: examine,example)",
    )

    parser.add_argument(
        "--compress",
        choices=["gzip", "bz2", "xz"],
        help="Compression type for output file",
    )

    parser.add_argument(
        "--report", action="store_true", help="Generate detailed summary report"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually concatenating",
    )

    args = parser.parse_args()

    # Parse exclude directories
    exclude_dirs = (
        [d.strip() for d in args.exclude_dirs.split(",") if d.strip()]
        if args.exclude_dirs
        else []
    )

    try:
        # Find all study files
        logger.info("🔍 Finding study results files...")
        study_files = find_study_files(args.input_dir, exclude_dirs)

        if not study_files:
            logger.error("No study CSV files found!")
            return 1

        # Show what would be processed
        logger.info(f"Found {len(study_files)} studies to process:")
        for i, csv_file in enumerate(study_files, 1):
            study_id = csv_file.stem.replace("_counts", "")
            logger.info(f"  {i:2d}. {study_id} ({csv_file.parent.name})")

        if args.dry_run:
            logger.info(
                "Dry run complete. Use without --dry-run to actually concatenate."
            )
            return 0

        # Load and transpose each study
        logger.info("📊 Loading and transposing studies...")
        study_dataframes = []

        for csv_file in study_files:
            transposed_df = load_and_transpose_study(csv_file)
            if transposed_df is not None:
                study_dataframes.append(transposed_df)

        if not study_dataframes:
            logger.error("No study files could be loaded successfully!")
            return 1

        logger.info(f"Successfully loaded {len(study_dataframes)} studies")

        # Concatenate all studies
        logger.info("🔗 Concatenating all studies...")
        master_df = concatenate_studies(study_dataframes)

        # Generate report if requested
        if args.report:
            report = generate_summary_report(master_df, study_files)
            print(report)

            # Save report to file
            report_file = args.output.replace(".csv", "_report.txt")
            with open(report_file, "w") as f:
                f.write(report)
            logger.info(f"Summary report saved to: {report_file}")

        # Save master file
        logger.info("💾 Saving master file...")
        save_master_file(master_df, args.output, args.compress)

        logger.info("✅ Concatenation completed successfully!")
        logger.info(
            f"Final dataset: {master_df.shape[0]:,} samples × "
            f"{master_df.shape[1] - 1:,} genes"
        )

        return 0

    except Exception as e:
        logger.error(f"Error during concatenation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
