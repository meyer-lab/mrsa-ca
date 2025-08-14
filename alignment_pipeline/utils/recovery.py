#!/usr/bin/env python3
"""
Recovery utilities for handling missing/failed SRRs and samples.
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def get_completed_srrs(output_dir):
    """
    Get list of SRRs that have already been successfully processed.
    """
    completed_srrs = []

    try:
        for filename in os.listdir(output_dir):
            if filename.endswith("_result.csv") and not filename.startswith("GSM"):
                srr_id = filename.replace("_result.csv", "")
                result_path = os.path.join(output_dir, filename)

                # Verify the file is readable and non-empty
                try:
                    result_df = pd.read_csv(result_path, index_col=0)
                    if not result_df.empty:
                        completed_srrs.append(srr_id)
                except Exception:
                    logger.warning(
                        f"⚠️ Invalid result file for {srr_id}, will reprocess"
                    )
                    try:
                        os.remove(result_path)
                    except Exception:
                        pass

    except Exception as e:
        logger.warning(f"Error checking completed SRRs: {e}")

    if completed_srrs:
        logger.info(f"📊 Found {len(completed_srrs)} completed SRRs")

    return completed_srrs


def check_missing_samples(study_id, sample_to_srrs, output_dir):
    """
    Check which samples are missing from the final results file.
    Returns (samples_to_process, already_complete).
    """
    if not study_id:
        return sample_to_srrs, set()

    final_path = os.path.join(output_dir, f"{study_id}_counts.csv")

    if not os.path.exists(final_path):
        logger.info(f"📂 No final results file found ({study_id}_counts.csv)")
        logger.info(f"   Will process all {len(sample_to_srrs)} samples")
        return sample_to_srrs, set()

    try:
        logger.info(f"🔍 Found existing results: {study_id}_counts.csv")
        existing_results = pd.read_csv(final_path, index_col=0)
        existing_samples = set(existing_results.columns)
        expected_samples = set(sample_to_srrs.keys())

        already_complete = existing_samples.intersection(expected_samples)
        missing_samples = expected_samples - existing_samples

        logger.info("📊 Recovery analysis:")
        logger.info(f"   Expected: {len(expected_samples)} samples")
        logger.info(f"   Complete: {len(already_complete)} samples")
        logger.info(f"   Missing: {len(missing_samples)} samples")

        if not missing_samples:
            logger.info("✅ All samples already complete")
            return {}, already_complete

        # Return only the missing samples for processing
        samples_to_process = {
            sample_id: sample_to_srrs[sample_id] for sample_id in missing_samples
        }

        return samples_to_process, already_complete

    except Exception as e:
        logger.warning(f"⚠️ Could not load existing results: {e}")
        logger.info("   Will process all samples from scratch")
        return sample_to_srrs, set()


def validate_srr_results(sample_to_srrs, output_dir, missing_action="error"):
    """
    Validate that all expected SRR result files are present.
    Returns (all_present, missing_srrs).
    """
    missing_srrs = set()

    for sample_id, srr_list in sample_to_srrs.items():
        for srr in srr_list:
            result_path = os.path.join(output_dir, f"{srr}_result.csv")
            if not os.path.exists(result_path):
                missing_srrs.add(srr)
            else:
                # Check if file is readable and non-empty
                try:
                    result_df = pd.read_csv(result_path, index_col=0)
                    if result_df.empty:
                        missing_srrs.add(srr)
                except Exception:
                    missing_srrs.add(srr)

    all_present = len(missing_srrs) == 0
    total_expected = sum(len(srr_list) for srr_list in sample_to_srrs.values())

    if all_present:
        logger.info(f"✅ All {total_expected} SRR result files are present")
    else:
        logger.warning(
            f"⚠️ Missing {len(missing_srrs)} out of {total_expected} SRR files"
        )

        if missing_action == "error":
            raise ValueError(f"Missing {len(missing_srrs)} SRR result files")
        elif missing_action == "warn":
            logger.warning("Proceeding with missing SRRs (partial data)")
        elif missing_action == "recover":
            logger.info("🔄 Will attempt to recover missing SRRs")

    return all_present, missing_srrs


def recover_missing_srrs(
    missing_srrs,
    input_dir,
    output_dir,
    genome="homo_sapiens",
    return_type="gene",
    identifier="symbol",
    max_retries_per_srr=3,
):
    """
    Recover missing SRRs by downloading and aligning them.

    STRICT MODE: All SRRs must be successfully recovered or pipeline fails.

    Args:
        missing_srrs: Set of SRR IDs that need recovery
        input_dir: Directory for FASTQ files
        output_dir: Directory for result files
        genome: Genome to align against
        return_type: Type of results to return
        identifier: Gene identifier type
        max_retries_per_srr: Maximum retry attempts per SRR

    Returns:
        recovered_count: Number of SRRs successfully recovered

    Raises:
        ValueError: If any SRRs cannot be recovered after all retry attempts
    """
    if not missing_srrs:
        return 0

    logger.info(
        f"🔄 STRICT RECOVERY: Attempting to recover {len(missing_srrs)} missing SRRs"
    )
    logger.warning(
        "⚠️  ZERO TOLERANCE MODE: ALL SRRs must be recovered or pipeline will fail"
    )

    # Import core processing functions (avoid circular imports)
    from core.processing import align_srr, download_srr

    recovered_count = 0
    unrecoverable_srrs = []

    for srr in missing_srrs:
        logger.info(f"🔄 STRICT RECOVERY: Processing {srr}")

        recovery_success = False
        last_error = None

        for attempt in range(max_retries_per_srr):
            try:
                logger.info(
                    f"🔄 Recovery attempt {attempt + 1}/{max_retries_per_srr} for {srr}"
                )

                # Download first
                download_success = download_srr(srr, input_dir)
                if not download_success:
                    raise ValueError(f"Download failed for {srr}")

                # Then align
                result = align_srr(
                    srr,
                    input_dir,
                    output_dir,
                    genome,
                    return_type,
                    identifier,
                    cleanup_after=True,
                )

                if result is not None and not result.empty:
                    recovered_count += 1
                    logger.info(
                        f"✅ STRICT RECOVERY SUCCESS: {srr} recovered successfully"
                    )
                    recovery_success = True
                    break
                else:
                    raise ValueError(f"Alignment produced no results for {srr}")

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"⚠️  Recovery attempt {attempt + 1} failed for {srr}: {e}"
                )

                if attempt < max_retries_per_srr - 1:
                    logger.info(
                        f"🔄 Retrying {srr} (attempt {attempt + 2}/{max_retries_per_srr})"
                    )
                    # Add small delay between retries
                    import time

                    time.sleep(5)

        if not recovery_success:
            logger.error(
                f"❌ CRITICAL: Failed to recover {srr} after {max_retries_per_srr} attempts"
            )
            logger.error(f"   Last error: {last_error}")
            unrecoverable_srrs.append(srr)

    # STRICT EVALUATION
    total_missing = len(missing_srrs)
    logger.info(
        f"🔄 STRICT RECOVERY COMPLETE: {recovered_count}/{total_missing} SRRs recovered"
    )

    if unrecoverable_srrs:
        logger.error(
            f"❌ CRITICAL FAILURE: {len(unrecoverable_srrs)} SRRs could not be recovered"
        )
        logger.error("📋 UNRECOVERABLE SRRs requiring human intervention:")
        for srr in sorted(unrecoverable_srrs):
            logger.error(f"   • {srr}")

        logger.error("🚨 PIPELINE TERMINATION: Cannot proceed without complete dataset")
        raise ValueError(
            f"Recovery failed for {len(unrecoverable_srrs)} SRRs: {sorted(unrecoverable_srrs)}"
        )

    logger.info("✅ STRICT RECOVERY SUCCESS: All missing SRRs recovered successfully")
    return recovered_count
