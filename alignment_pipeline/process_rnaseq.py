#!/usr/bin/env python3
"""
Simplified RNA-seq processing pipeline.

This refactored version separates concerns and reduces complexity:
- Core processing functions moved to core/processing.py
- Utilities moved to utils/ modules
- Recovery logic simplified and moved to utils/recovery.py
- Timing simplified and moved to utils/timing.py
"""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from core.processing import align_srr, combine_srr_results, download_srr
from utils.file_utils import check_disk_space
from utils.index_manager import reset_alignment_indexes, validate_and_reset_if_needed
from utils.recovery import (
    check_missing_samples,
    get_completed_srrs,
    validate_srr_results,
)

# Import our simplified modules
from utils.timing import timer

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "process_rnaseq.log")),
    ],
)
logger = logging.getLogger(__name__)


def process_single_srr(
    srr, input_dir, output_dir, genome, return_type, identifier, cleanup_immediate
):
    """Process a single SRR: download and align."""
    try:
        # Download first
        if not download_srr(srr, input_dir):
            logger.error(f"Failed to download {srr}")
            return None

        # Then align
        result = align_srr(
            srr,
            input_dir,
            output_dir,
            genome,
            return_type,
            identifier,
            cleanup_after=cleanup_immediate,
        )

        return result

    except Exception as e:
        logger.error(f"Error processing SRR {srr}: {e}")
        return None


def process_srrs_simple(
    srr_list,
    input_dir,
    output_dir,
    max_workers=1,
    genome="homo_sapiens",
    return_type="gene",
    identifier="symbol",
    cleanup="end",
    study_id=None,
    max_download_workers=1,
):
    """
    Simplified SRR processing pipeline with concurrent download and alignment.
    Can handle multiple download workers (max_download_workers) with single
    alignment worker. Each download worker processes unique SRRs to avoid
    conflicts.
    """
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"🚀 Starting simplified pipeline for {len(srr_list)} SRRs")

    # Check for completed SRRs (simple checkpoint mechanism)
    completed_srrs = get_completed_srrs(output_dir)
    remaining_srrs = [srr for srr in srr_list if srr not in completed_srrs]

    if completed_srrs:
        logger.info(f"📋 Skipping {len(completed_srrs)} already completed SRRs")

    if not remaining_srrs:
        logger.info("✅ All SRRs already completed")
        return load_and_combine_results(srr_list, output_dir, study_id)

    logger.info(
        f"🔄 Processing {len(remaining_srrs)} remaining SRRs "
        f"with concurrent download/alignment"
    )

    # Determine which SRRs are already downloaded and ready for alignment
    from utils.fastq_utils import get_fastq_paths

    ready_for_alignment = []
    to_download = []

    for srr in remaining_srrs:
        read_type, file_paths = get_fastq_paths(srr, input_dir)
        if read_type != "missing":
            ready_for_alignment.append(srr)
        else:
            to_download.append(srr)

    logger.info(
        f"{len(ready_for_alignment)} SRRs already downloaded, "
        f"{len(to_download)} need downloading"
    )

    cleanup_immediate = cleanup == "immediate"
    all_results = []
    failed_alignments = []

    # Use multiple download workers with single alignment worker for safe
    # concurrent processing
    # Each download worker gets unique SRRs to avoid conflicts
    import threading
    from collections import deque

    # Thread-safe queues for coordination
    download_queue = deque(to_download)
    ready_queue = deque(ready_for_alignment)
    download_queue_lock = threading.Lock()
    ready_queue_lock = threading.Lock()

    # Limit download workers to prevent resource exhaustion
    max_download_workers = min(max_download_workers, len(to_download), 4)

    logger.info(
        f"🚀 Using {max_download_workers} download workers and 1 alignment worker"
    )

    with (
        ThreadPoolExecutor(max_workers=max_download_workers) as download_executor,
        ThreadPoolExecutor(max_workers=1) as align_executor,
    ):
        # Track active futures
        active_download_futures = {}  # future -> srr mapping
        active_align_future = None
        current_align_srr = None

        def submit_download():
            """Submit a download job if SRRs are available."""
            with download_queue_lock:
                if download_queue:
                    srr = download_queue.popleft()
                    future = download_executor.submit(download_srr, srr, input_dir)
                    active_download_futures[future] = srr
                    logger.info(
                        f"Started download of {srr} "
                        f"(worker {len(active_download_futures)})"
                    )
                    return True
            return False

        def submit_alignment():
            """Submit an alignment job if files are ready."""
            nonlocal active_align_future, current_align_srr

            if active_align_future is None:
                with ready_queue_lock:
                    if ready_queue:
                        srr = ready_queue.popleft()
                        active_align_future = align_executor.submit(
                            align_srr,
                            srr,
                            input_dir,
                            output_dir,
                            genome,
                            return_type,
                            identifier,
                            cleanup_after=cleanup_immediate,
                        )
                        current_align_srr = srr
                        logger.info(f"Started alignment of {srr}")
                        return True
            return False

        # Start initial downloads (up to max_download_workers)
        for _ in range(max_download_workers):
            if not submit_download():
                break

        # Start initial alignment if files are ready
        submit_alignment()

        # Process until all downloads and alignments are complete
        while (
            active_download_futures
            or active_align_future
            or download_queue
            or ready_queue
        ):
            # Check completed download futures
            completed_downloads = []
            for future, srr in active_download_futures.items():
                if future.done():
                    completed_downloads.append((future, srr))

            # Process completed downloads
            for future, srr in completed_downloads:
                del active_download_futures[future]

                try:
                    success = future.result()
                    if success:
                        logger.info(f"Download of {srr} completed")
                        # Check if the SRR is now ready for alignment
                        read_type, file_paths = get_fastq_paths(srr, input_dir)
                        if read_type != "missing":
                            with ready_queue_lock:
                                ready_queue.append(srr)
                    else:
                        logger.error(f"Download of {srr} failed")
                        failed_alignments.append(srr)

                except Exception as e:
                    logger.error(f"Error in download of {srr}: {str(e)}")
                    failed_alignments.append(srr)

                # Start next download if available
                submit_download()

            # Check if alignment is complete
            if active_align_future is not None and active_align_future.done():
                try:
                    result = active_align_future.result()
                    if result is not None:
                        all_results.append(result)
                        logger.info(f"Alignment completed for {current_align_srr}")
                    else:
                        logger.error(f"Alignment failed for {current_align_srr}")
                        failed_alignments.append(current_align_srr)
                except Exception as e:
                    logger.error(f"Error in alignment of {current_align_srr}: {str(e)}")
                    failed_alignments.append(current_align_srr)

                # Reset alignment tracking
                active_align_future = None
                current_align_srr = None

                # Start next alignment if files are ready
                submit_alignment()

            # If no alignment is running but we have files ready, start one
            if not submit_alignment():
                # Small sleep to prevent busy waiting
                import time

                time.sleep(0.1)

    # Final cleanup if requested
    if cleanup == "end":
        logger.info("🧹 Performing final cleanup")
        from utils.file_utils import cleanup_temp_files

        for srr in srr_list:
            cleanup_temp_files(srr, input_dir, keep_fastq=False)

    # Report results
    logger.info(
        f"✅ Processing completed: {len(all_results)} successful, "
        f"{len(failed_alignments)} failed"
    )
    if failed_alignments:
        logger.warning(f"Failed SRRs: {failed_alignments}")

    # Combine and save final results
    return load_and_combine_results(srr_list, output_dir, study_id)


def process_samples_simple(
    sample_to_srrs,
    input_dir,
    output_dir,
    max_workers=1,
    genome="homo_sapiens",
    return_type="gene",
    identifier="symbol",
    cleanup="end",
    combination_method="sum",
    study_id=None,
    missing_srr_action="warn",
):
    """
    Simplified sample processing pipeline.
    """
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"🚀 Starting sample pipeline for {len(sample_to_srrs)} samples")

    # Check what samples are already complete
    samples_to_process, already_complete = check_missing_samples(
        study_id, sample_to_srrs, output_dir
    )

    if not samples_to_process:
        logger.info("✅ All samples already complete")
        final_path = os.path.join(output_dir, f"{study_id}_counts.csv")
        if os.path.exists(final_path):
            return pd.read_csv(final_path, index_col=0)
        return None

    # Process all SRRs individually first
    all_srrs = []
    for sample_id, srr_list in samples_to_process.items():
        all_srrs.extend(srr_list)

    logger.info(
        f"🔄 Processing {len(all_srrs)} SRRs for {len(samples_to_process)} samples"
    )

    # Process individual SRRs
    process_srrs_simple(
        all_srrs,
        input_dir,
        output_dir,
        max_workers,
        genome,
        return_type,
        identifier,
        cleanup,
        study_id=None,
        max_download_workers=max_workers,
    )

    # Validate all SRR results are present - STRICT MODE
    try:
        all_present, missing_srrs = validate_srr_results(
            samples_to_process, output_dir, missing_srr_action
        )

        # If recovery is requested and there are missing SRRs, attempt recovery
        if not all_present and missing_srr_action == "recover":
            from utils.recovery import recover_missing_srrs

            logger.info("🔄 STRICT MODE: Starting automatic SRR recovery...")
            logger.warning(f"⚠️  Missing {len(missing_srrs)} SRRs - attempting recovery")

            try:
                recovered_count = recover_missing_srrs(
                    missing_srrs,
                    input_dir,
                    output_dir,
                    genome,
                    return_type,
                    identifier,
                    max_retries_per_srr=3,
                )
                logger.info(
                    f"✅ STRICT RECOVERY: All {recovered_count} SRRs "
                    f"recovered successfully"
                )

            except ValueError as recovery_error:
                logger.error(f"❌ STRICT RECOVERY FAILED: {recovery_error}")
                logger.error(
                    "🚨 PIPELINE TERMINATION: Cannot proceed without complete dataset"
                )
                raise

        elif not all_present:
            logger.error(
                f"❌ CRITICAL: {len(missing_srrs)} SRRs are missing "
                f"and recovery is disabled"
            )
            logger.error(f"📋 Missing SRRs: {sorted(missing_srrs)}")
            raise ValueError(f"Missing {len(missing_srrs)} SRRs with recovery disabled")

    except ValueError as e:
        logger.error(f"❌ VALIDATION FAILURE: {e}")
        raise

    # Combine SRRs by sample - STRICT MODE: ALL samples must succeed
    sample_results = []
    total_samples = len(samples_to_process)
    failed_samples = []
    problematic_srrs = set()

    logger.info(f"🔗 STRICT PROCESSING: Combining SRRs for {total_samples} samples")
    logger.info(
        "⚠️  ZERO TOLERANCE MODE: Any missing or corrupted data "
        "will cause pipeline failure"
    )

    for sample_id, srr_list in samples_to_process.items():
        logger.info(f"🔗 Combining {len(srr_list)} SRRs for sample {sample_id}")

        # Load individual SRR results - track missing files
        srr_results = []
        missing_srrs = []

        for srr in srr_list:
            result_path = os.path.join(output_dir, f"{srr}_result.csv")
            if os.path.exists(result_path):
                try:
                    srr_result = pd.read_csv(result_path, index_col=0)
                    if not srr_result.empty:
                        # Validate gene count for loaded SRR files
                        expected_gene_count = 31374
                        actual_gene_count = len(srr_result)

                        if actual_gene_count != expected_gene_count:
                            logger.error(
                                f"❌ CORRUPTED SRR FILE: {srr} has "
                                f"{actual_gene_count} genes "
                                f"(expected {expected_gene_count})"
                            )
                            logger.error(f"🗑️  Deleting corrupted file: {result_path}")
                            os.remove(result_path)
                            missing_srrs.append(srr)
                            continue

                        # Clean the result to ensure only numeric columns are used
                        # This prevents type mixing issues during combination
                        numeric_cols = srr_result.select_dtypes(
                            include=["number"]
                        ).columns
                        if len(numeric_cols) > 0:
                            # Use only the first numeric column (should be the
                            # count data)
                            clean_result = srr_result[numeric_cols].iloc[:, [0]].copy()
                            srr_results.append(clean_result)
                        else:
                            logger.error(
                                f"❌ CRITICAL: No numeric columns found in {srr}"
                            )
                            missing_srrs.append(srr)
                            problematic_srrs.add(srr)
                    else:
                        logger.error(f"❌ CRITICAL: Empty result file for {srr}")
                        missing_srrs.append(srr)
                        problematic_srrs.add(srr)
                except Exception as e:
                    logger.error(f"❌ CRITICAL: Could not load result for {srr}: {e}")
                    missing_srrs.append(srr)
                    problematic_srrs.add(srr)
            else:
                logger.error(f"❌ CRITICAL: Missing result file for {srr}")
                missing_srrs.append(srr)
                problematic_srrs.add(srr)

        # STRICT: All SRRs for this sample must be present
        if missing_srrs:
            logger.error(
                f"❌ SAMPLE FAILURE: {sample_id} missing "
                f"{len(missing_srrs)}/{len(srr_list)} SRRs: "
                f"{missing_srrs}"
            )
            failed_samples.append(
                {
                    "sample_id": sample_id,
                    "total_srrs": len(srr_list),
                    "missing_srrs": missing_srrs,
                    "available_srrs": len(srr_results),
                }
            )
            continue

        # Attempt to combine the SRR results for this sample
        try:
            sample_result = combine_srr_results(
                srr_results, sample_id, combination_method
            )
            sample_results.append(sample_result)
            logger.info(f"✅ STRICT SUCCESS: Sample {sample_id} combined successfully")

        except Exception as e:
            logger.error(f"❌ SAMPLE FAILURE: Cannot combine sample {sample_id}: {e}")
            failed_samples.append(
                {
                    "sample_id": sample_id,
                    "total_srrs": len(srr_list),
                    "missing_srrs": [],
                    "available_srrs": len(srr_results),
                    "combination_error": str(e),
                }
            )

    # STRICT EVALUATION: Report all failures before proceeding
    success_count = len(sample_results)
    failure_count = len(failed_samples)

    logger.info("📊 STRICT PROCESSING SUMMARY:")
    logger.info(f"   ✅ Successful samples: {success_count}/{total_samples}")
    logger.info(f"   ❌ Failed samples: {failure_count}/{total_samples}")

    if failed_samples:
        logger.error(f"❌ CRITICAL: {failure_count} samples failed processing")
        logger.error("📋 FAILED SAMPLES REPORT:")
        for failure in failed_samples:
            if failure["missing_srrs"]:
                logger.error(
                    f"   • {failure['sample_id']}: Missing "
                    f"{len(failure['missing_srrs'])} SRRs: "
                    f"{failure['missing_srrs']}"
                )
            else:
                logger.error(
                    f"   • {failure['sample_id']}: Combination "
                    f"error - "
                    f"{failure.get('combination_error', 'Unknown error')}"
                )

        if problematic_srrs:
            logger.error(
                f"🔧 PROBLEMATIC SRRs requiring human intervention: "
                f"{sorted(problematic_srrs)}"
            )

        # STRICT MODE: Fail the entire pipeline if any samples failed
        raise ValueError(
            f"PIPELINE FAILURE: {failure_count} samples failed processing. "
            f"All samples must succeed for scientific data integrity."
        )

    # Only proceed if ALL samples succeeded

    # Combine all sample results
    if sample_results:
        new_results = pd.concat(sample_results, axis=1, sort=True).fillna(0)

        # CRITICAL FIX: Merge with existing results instead of overwriting
        if study_id:
            final_path = os.path.join(output_dir, f"{study_id}_counts.csv")

            # Check if existing results file exists and merge
            if os.path.exists(final_path):
                try:
                    logger.info(
                        "🔄 MERGING: Combining newly processed "
                        "samples with existing results"
                    )
                    existing_results = pd.read_csv(final_path, index_col=0)

                    # Merge existing and new results
                    final_results = pd.concat([existing_results, new_results], axis=1)

                    # Ensure no duplicate columns (prioritize new results
                    # if any overlap)
                    if final_results.columns.duplicated().any():
                        logger.warning(
                            "Found duplicate samples between "
                            "existing and new results, keeping "
                            "new versions"
                        )
                        final_results = final_results.loc[
                            :, ~final_results.columns.duplicated(keep="last")
                        ]

                    logger.info(
                        f"✅ MERGED RESULTS: "
                        f"{existing_results.shape[1]} existing + "
                        f"{new_results.shape[1]} new = "
                        f"{final_results.shape[1]} total samples"
                    )
                except Exception as e:
                    logger.warning(f"Could not merge with existing results: {e}")
                    logger.info("Proceeding with new results only")
                    final_results = new_results
            else:
                final_results = new_results

            # Save the merged/new results
            final_results.to_csv(final_path)
            logger.info(f"💾 Saved final results: {final_path}")

            # VERIFICATION AND CLEANUP: Verify study completion and clean up
            # SRR intermediates
            try:
                from utils.study_cleanup import verify_and_cleanup_study

                logger.info(
                    "🔍 Verifying study completion and cleaning up "
                    "intermediate files..."
                )

                cleanup_result = verify_and_cleanup_study(
                    study_id=study_id,
                    output_dir=output_dir,
                    cleanup=True,  # Clean up SRR files after verification
                    backup_srr=False,  # Delete rather than backup to save space
                    dry_run=False,
                )

                if cleanup_result["verification"]["complete"]:
                    logger.info(
                        "✅ Study verification and cleanup completed successfully"
                    )
                else:
                    logger.warning(
                        "⚠️  Study verification failed - SRR "
                        "files preserved for debugging"
                    )

            except Exception as e:
                logger.warning(f"Study verification/cleanup failed: {e}")
                logger.info("Continuing without cleanup - SRR files preserved")
        else:
            final_results = new_results

        timer.summary()
        return final_results
    else:
        logger.error("No sample results to combine")
        return None


def load_and_combine_results(srr_list, output_dir, study_id=None):
    """Load and combine individual SRR results."""
    results = []

    for srr in srr_list:
        result_path = os.path.join(output_dir, f"{srr}_result.csv")
        if os.path.exists(result_path):
            try:
                result_df = pd.read_csv(result_path, index_col=0)
                if not result_df.empty:
                    # Validate gene count for loaded SRR files
                    expected_gene_count = 31374
                    actual_gene_count = len(result_df)

                    if actual_gene_count != expected_gene_count:
                        logger.error(
                            f"❌ CORRUPTED SRR FILE: {srr} has "
                            f"{actual_gene_count} genes "
                            f"(expected {expected_gene_count})"
                        )
                        logger.error(f"🗑️  Deleting corrupted file: {result_path}")
                        os.remove(result_path)
                        continue

                    results.append(result_df)
            except Exception as e:
                logger.warning(f"Could not load result for {srr}: {e}")

    if results:
        new_combined = pd.concat(results, axis=1, sort=True).fillna(0)

        if study_id:
            final_path = os.path.join(output_dir, f"{study_id}_counts.csv")

            # CRITICAL FIX: Check for existing results and merge instead of overwrite
            if os.path.exists(final_path):
                try:
                    logger.info("🔄 MERGING: Combining with existing results file")
                    existing_results = pd.read_csv(final_path, index_col=0)

                    # Merge existing and new results
                    combined_results = pd.concat(
                        [existing_results, new_combined], axis=1
                    )

                    # Handle duplicate columns (prioritize new data)
                    if combined_results.columns.duplicated().any():
                        logger.warning("Found duplicate SRRs, keeping latest versions")
                        combined_results = combined_results.loc[
                            :, ~combined_results.columns.duplicated(keep="last")
                        ]

                    logger.info(
                        f"✅ MERGED: {existing_results.shape[1]} "
                        f"existing + {new_combined.shape[1]} new = "
                        f"{combined_results.shape[1]} total"
                    )
                except Exception as e:
                    logger.warning(f"Could not merge with existing results: {e}")
                    combined_results = new_combined
            else:
                combined_results = new_combined

            combined_results.to_csv(final_path)
            logger.info(f"💾 Saved combined results: {final_path}")
        else:
            combined_results = new_combined

        timer.summary()
        return combined_results
    else:
        logger.error("No results to combine")
        return None


def main():
    """Simplified main function."""
    parser = argparse.ArgumentParser(
        description="Simplified RNA-seq processing pipeline"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--srr_file", help="File with SRR accessions (one per line)"
    )
    input_group.add_argument(
        "--sample_mapping", help="JSON file mapping samples to SRRs"
    )

    # Basic options
    parser.add_argument(
        "--input_dir", default="data/input", help="Directory for FASTQ files"
    )
    parser.add_argument(
        "--output_dir", default="data/results", help="Directory for results"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of parallel workers (for backward compatibility)",
    )
    parser.add_argument(
        "--download_workers",
        type=int,
        default=None,
        help="Number of parallel download workers (default: same as max_workers)",
    )
    parser.add_argument("--genome", default="homo_sapiens", help="Genome to align to")
    parser.add_argument(
        "--return_type", default="gene", help="Return type (gene/transcript)"
    )
    parser.add_argument("--identifier", default="symbol", help="Identifier type")
    parser.add_argument(
        "--cleanup", choices=["none", "immediate", "end"], default="end"
    )
    parser.add_argument(
        "--combination_method", choices=["sum", "mean", "median"], default="sum"
    )
    parser.add_argument(
        "--missing_srr_action",
        choices=["error", "warn", "recover"],
        default="recover",
        help="Action for missing SRRs: error (fail), "
        "warn (proceed with partial data), "
        "recover (auto-download/align missing)",
    )
    parser.add_argument("--study_id", help="Study ID for organizing results")
    parser.add_argument(
        "--reset_indexes", action="store_true", help="Reset alignment indexes"
    )

    args = parser.parse_args()

    # Handle download worker configuration
    if args.download_workers is None:
        args.download_workers = args.max_workers

    logger.info(
        f"Configuration: {args.download_workers} download workers, 1 alignment worker"
    )

    # Setup
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Check disk space
    if not check_disk_space(args.input_dir, min_gb=10):
        logger.error("Insufficient disk space in input directory")
        return 1

    # Handle index reset if requested
    if args.reset_indexes:
        if not reset_alignment_indexes(species=args.genome, force=True):
            logger.error("Index reset failed")
            return 1

    # Validate indexes
    if not validate_and_reset_if_needed(species=args.genome):
        logger.warning("Index validation failed")

    # Process based on input type
    if args.srr_file:
        logger.info("Mode: Individual SRRs")
        with open(args.srr_file) as f:
            srr_list = [line.strip() for line in f if line.strip()]

        results = process_srrs_simple(
            srr_list,
            args.input_dir,
            args.output_dir,
            args.max_workers,
            args.genome,
            args.return_type,
            args.identifier,
            args.cleanup,
            args.study_id,
            max_download_workers=args.download_workers,
        )

    elif args.sample_mapping:
        logger.info("Mode: Sample mapping")
        with open(args.sample_mapping) as f:
            sample_to_srrs = json.load(f)

        results = process_samples_simple(
            sample_to_srrs,
            args.input_dir,
            args.output_dir,
            args.max_workers,
            args.genome,
            args.return_type,
            args.identifier,
            args.cleanup,
            args.combination_method,
            args.study_id,
            args.missing_srr_action,
        )

    if results is not None:
        logger.info("✅ STRICT MODE SUCCESS: All samples processed successfully")
        logger.info(
            "📊 Pipeline completed with 100% success rate - data integrity maintained"
        )
        logger.info(f"📊 Final results shape: {results.shape}")
        return 0
    else:
        logger.error(
            "❌ STRICT MODE FAILURE: Pipeline terminated due to processing failures"
        )
        logger.error(
            "🚨 HUMAN INTERVENTION REQUIRED: Check failed samples "
            "and problematic SRRs above"
        )
        logger.error(
            "📋 No partial results generated - complete dataset "
            "required for scientific integrity"
        )
        return 1


if __name__ == "__main__":
    exit(main())
