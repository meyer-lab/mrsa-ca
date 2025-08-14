#!/usr/bin/env python3
"""
Core processing functions for SRR download and alignment.
"""

import logging
import os

import pandas as pd
import xalign
from utils.fastq_utils import get_fastq_paths, validate_fastq_files
from utils.file_utils import cleanup_temp_files, save_csv_result_securely
from utils.timing import timer

logger = logging.getLogger(__name__)


def download_srr(srr, input_dir, max_retries=3):
    """Download an SRR using fasterq-dump with same parameters as xalign."""
    logger.info(f"🔄 Downloading {srr}")
    timer.start("download", srr)

    # Check if files already exist
    single_path = os.path.join(input_dir, f"{srr}.fastq")
    paired_path1 = os.path.join(input_dir, f"{srr}_1.fastq")
    paired_path2 = os.path.join(input_dir, f"{srr}_2.fastq")

    if os.path.exists(single_path) or (
        os.path.exists(paired_path1) and os.path.exists(paired_path2)
    ):
        logger.info(f"✅ {srr} already downloaded")
        timer.end("download", srr)
        return True

    # Create lock file to prevent concurrent downloads
    lock_path = os.path.join(input_dir, f"{srr}.lock")
    try:
        with open(lock_path, "w") as f:
            f.write(f"Download started at {timer.pipeline_start}")

        # Get thread allocation from environment (set by run_quantify.sh)
        download_threads = int(os.environ.get("PIPELINE_DOWNLOAD_THREADS", "2"))

        # Use fasterq-dump with exact parameters as xalign.sra.load_sra()
        import subprocess

        cmd = [
            "fasterq-dump",
            srr,
            "-f",  # force overwrite
            "--mem",
            "2G",  # memory limit
            "--split-3",  # split paired-end reads
            "--threads",
            str(download_threads),  # thread count
            "--skip-technical",  # skip technical reads
            "-O",
            input_dir,  # output directory
            "--temp",
            input_dir,  # temp directory to scratch (CRITICAL for preventing home dir overflow)
        ]

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Download attempt {attempt + 1}/{max_retries} for {srr} with {download_threads} threads"
                )
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=3600
                )

                if result.returncode == 0:
                    logger.info(f"✅ {srr} downloaded successfully")
                    timer.end("download", srr)
                    return True
                else:
                    logger.warning(
                        f"Download attempt {attempt + 1}/{max_retries} failed for {srr}: {result.stderr}"
                    )
                    if attempt < max_retries - 1:
                        import time

                        wait_time = 5 * (
                            attempt + 1
                        )  # Progressive backoff: 5s, 10s, 15s
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)

            except subprocess.TimeoutExpired:
                logger.error(
                    f"Download attempt {attempt + 1}/{max_retries} timeout for {srr}"
                )
                if attempt < max_retries - 1:
                    import time

                    logger.info("Waiting 10s before retry after timeout...")
                    time.sleep(10)
            except Exception as e:
                logger.error(
                    f"Download attempt {attempt + 1}/{max_retries} error for {srr}: {e}"
                )
                if attempt < max_retries - 1:
                    import time

                    logger.info("Waiting 5s before retry after error...")
                    time.sleep(5)

        timer.end("download", srr)
        return False

    finally:
        # Clean up lock file
        if os.path.exists(lock_path):
            os.remove(lock_path)


def align_srr(
    srr,
    input_dir,
    output_dir,
    genome="homo_sapiens",
    return_type="gene",
    identifier="symbol",
    max_retries=3,
    cleanup_after=False,
):
    """Align an SRR using xalign."""
    logger.info(f"🧬 Aligning {srr}")
    timer.start("alignment", srr)

    result_path = os.path.join(output_dir, f"{srr}_result.csv")

    # Skip if result already exists
    if os.path.exists(result_path):
        try:
            result_df = pd.read_csv(result_path, index_col=0)
            if not result_df.empty:
                # Validate gene count for existing files
                expected_gene_count = 31374
                actual_gene_count = len(result_df)

                if actual_gene_count == expected_gene_count:
                    logger.info(f"✅ {srr} already aligned ({actual_gene_count} genes)")
                    timer.end("alignment", srr)
                    return result_df
                else:
                    logger.warning(
                        f"🚨 Existing {srr} has wrong gene count: {actual_gene_count} != {expected_gene_count}"
                    )
                    logger.warning(f"Will delete and re-align {srr}")
                    os.remove(result_path)
        except Exception as e:
            logger.warning(
                f"Existing result file corrupt for {srr}, will re-align: {e}"
            )

    # Get FASTQ file paths
    read_type, file_paths = get_fastq_paths(srr, input_dir)

    if read_type == "missing":
        logger.error(f"No FASTQ files found for {srr}")
        timer.end("alignment", srr)
        return None

    if not validate_fastq_files(file_paths):
        logger.error(f"Invalid FASTQ files for {srr}")
        timer.end("alignment", srr)
        return None

    # Perform alignment with retries
    for attempt in range(max_retries):
        try:
            logger.info(f"Alignment attempt {attempt + 1}/{max_retries} for {srr}")

            # Get thread allocation from environment (set by run_quantify.sh)
            alignment_threads = int(os.environ.get("PIPELINE_ALIGNMENT_THREADS", "4"))

            if read_type == "single":
                result = xalign.align_fastq(
                    genome,
                    file_paths[0],  # Note: xalign expects (species, fastq) order
                    aligner="kallisto",
                    t=alignment_threads,
                    verbose=True,
                )
            else:  # paired
                result = xalign.align_fastq(
                    genome,
                    [
                        file_paths[0],
                        file_paths[1],
                    ],  # Note: xalign expects (species, fastq) order
                    aligner="kallisto",
                    t=alignment_threads,
                    verbose=True,
                )

            if result is not None and not result.empty:
                # Process the result into the requested format
                result_df = process_alignment_result(
                    result, srr, return_type, identifier
                )

                if result_df is not None and not result_df.empty:
                    if save_csv_result_securely(result_df, result_path):
                        logger.info(f"✅ {srr} aligned successfully")

                        # Cleanup if requested
                        if cleanup_after:
                            cleanup_temp_files(srr, input_dir, keep_fastq=False)

                        timer.end("alignment", srr)
                        return result_df

            logger.warning(
                f"Alignment attempt {attempt + 1}/{max_retries} failed for {srr} - empty or null result"
            )
            if attempt < max_retries - 1:
                import time

                wait_time = 10 * (attempt + 1)  # Progressive backoff: 10s, 20s, 30s
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

        except Exception as e:
            logger.error(
                f"Alignment attempt {attempt + 1}/{max_retries} failed for {srr}: {e}"
            )
            if attempt < max_retries - 1:
                import time

                wait_time = 10 * (attempt + 1)  # Progressive backoff: 10s, 20s, 30s
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

    timer.end("alignment", srr)
    return None


def process_alignment_result(result, srr, return_type="gene", identifier="symbol"):
    """
    Process alignment result from xalign into the requested format.
    Based on the working legacy implementation.
    """
    if result is None:
        return None

    try:
        # xalign.align_fastq returns a DataFrame with columns: ['transcript', 'reads', 'tpm']
        if isinstance(result, pd.DataFrame):
            if return_type == "gene":
                # Convert transcript-level to gene-level counts using xalign's built-in aggregation
                try:
                    # Use xalign's gene aggregation function
                    import xalign.ensembl

                    gene_result = xalign.ensembl.agg_gene_counts(
                        result, "homo_sapiens", identifier=identifier
                    )
                    # gene_result should have columns like ['gene', 'counts']
                    if hasattr(gene_result, "iloc") and len(gene_result) > 0:
                        gene_result_df = pd.DataFrame(
                            gene_result["counts"].values,
                            index=gene_result.iloc[:, 0],
                            columns=[srr],
                        )
                        # Ensure integer type for storage efficiency (legacy requirement)
                        gene_result_df = gene_result_df.astype(int)
                        return gene_result_df
                    else:
                        raise ValueError("Gene aggregation returned empty result")
                except Exception as e:
                    logger.warning(
                        f"Gene aggregation failed for {srr}, using transcript counts: {e}"
                    )
                    # Fall back to transcript counts
                    result_df = pd.DataFrame(
                        result["reads"].values,
                        index=result["transcript"],
                        columns=[srr],
                    )
                    # Ensure integer type for storage efficiency (legacy requirement)
                    result_df = result_df.astype(int)
                    return result_df
            else:
                # Return transcript-level counts
                result_df = pd.DataFrame(
                    result["reads"].values, index=result["transcript"], columns=[srr]
                )
                # Ensure integer type for storage efficiency (legacy requirement)
                result_df = result_df.astype(int)
                return result_df
        else:
            logger.warning(f"Unexpected result type for {srr}: {type(result)}")
            return None

    except Exception as e:
        logger.error(f"Error processing alignment result for {srr}: {e}")
        return None


def combine_srr_results(srr_results, sample_id, combination_method="sum"):
    """
    Combine multiple SRR results into a single sample result.

    STRICT MODE: This function MUST succeed for all expected SRRs or fail completely.
    No partial results are acceptable for scientific data integrity.
    """
    valid_results = [r for r in srr_results if r is not None and not r.empty]

    if not valid_results:
        logger.error(f"❌ CRITICAL: No valid results found for sample {sample_id}")
        raise ValueError(
            f"No valid results for sample {sample_id} - pipeline cannot proceed"
        )

    try:
        # Clean and normalize the data before combining
        cleaned_results = []
        problematic_indices = []

        for i, df in enumerate(valid_results):
            # Select only numeric columns to avoid type mixing issues
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                # Use only the first numeric column which should be the count data
                clean_df = df[numeric_cols].iloc[:, [0]].copy()
                # Ensure integer type for storage efficiency (legacy requirement)
                try:
                    clean_df = clean_df.astype(int)
                    cleaned_results.append(clean_df)
                except ValueError as ve:
                    logger.error(
                        f"❌ CRITICAL: Cannot convert data to integer for sample {sample_id}, result {i}: {ve}"
                    )
                    problematic_indices.append(i)
            else:
                logger.error(
                    f"❌ CRITICAL: No numeric columns found in result {i} for sample {sample_id}"
                )
                problematic_indices.append(i)

        if problematic_indices:
            raise ValueError(
                f"Data type errors in sample {sample_id} at result indices: {problematic_indices}"
            )

        if not cleaned_results:
            logger.error(
                f"❌ CRITICAL: No valid numeric data found for sample {sample_id}"
            )
            raise ValueError(f"No valid numeric data for sample {sample_id}")

        # STRICT: All expected SRRs must be present
        if len(cleaned_results) != len(srr_results):
            missing_count = len(srr_results) - len(cleaned_results)
            logger.error(
                f"❌ CRITICAL: Missing {missing_count} SRR results for sample {sample_id}"
            )
            raise ValueError(
                f"Incomplete data for sample {sample_id}: expected {len(srr_results)}, got {len(cleaned_results)}"
            )

        # Combine all results
        combined_df = pd.concat(cleaned_results, axis=1, sort=True).fillna(0)

        # Apply combination method
        if combination_method == "sum":
            result_series = combined_df.sum(axis=1)
        elif combination_method == "mean":
            result_series = combined_df.mean(axis=1)
        elif combination_method == "median":
            result_series = combined_df.median(axis=1)
        else:
            logger.error(
                f"❌ CRITICAL: Unknown combination method: {combination_method}"
            )
            raise ValueError(f"Invalid combination method: {combination_method}")

        # Ensure integer type for final result (legacy requirement for storage efficiency)
        result_series = result_series.astype(int)
        sample_df = pd.DataFrame({sample_id: result_series})
        logger.info(
            f"✅ STRICT SUCCESS: Combined {len(cleaned_results)} SRRs for sample {sample_id}"
        )
        return sample_df

    except Exception as e:
        logger.error(
            f"❌ CRITICAL FAILURE: Cannot combine results for sample {sample_id}: {e}"
        )
        raise  # Re-raise to ensure pipeline fails
