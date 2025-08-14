#!/usr/bin/env python3
"""
File and system utilities for the RNA-seq processing pipeline.
"""

import logging
import os
import shutil
import time

import pandas as pd

logger = logging.getLogger(__name__)


def wait_for_file(file_path, timeout=600, check_interval=30, post_detection_wait=5):
    """Wait for a file to become fully available and not busy."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            try:
                # Try to open the file to see if it's accessible
                with open(file_path, "rb") as f:
                    f.read(1)  # Read a single byte

                # Wait a bit more to ensure the file is fully written
                time.sleep(post_detection_wait)
                logger.info(f"File ready: {file_path}")
                return True

            except (OSError, PermissionError):
                logger.debug(f"File exists but not ready: {file_path}")
                time.sleep(check_interval)
        else:
            time.sleep(check_interval)

    logger.error(f"Timeout waiting for file {file_path}")
    return False


def force_remove_file(file_path, max_retries=3, retry_delay=1):
    """
    Forcefully remove a file, handling permission and access issues.
    """
    if not os.path.exists(file_path):
        return True

    for attempt in range(max_retries):
        try:
            os.remove(file_path)
            logger.debug(f"Successfully removed file: {file_path}")
            return True
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} to remove {file_path} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Failed to remove {file_path} after {max_retries} attempts"
                )

    return False


def save_csv_result_securely(result_df, result_path, max_retries=3):
    """
    Save CSV result with atomic write operation to prevent corruption.
    Includes validation for expected gene count (31,374 genes).
    """
    if result_df is None or result_df.empty:
        logger.warning(f"Cannot save empty result to {result_path}")
        return False

    # Validate gene count - should be exactly 31,374 genes
    expected_gene_count = 31374
    actual_gene_count = len(result_df)

    if actual_gene_count != expected_gene_count:
        logger.error(
            f"❌ GENE COUNT VALIDATION FAILED for {os.path.basename(result_path)}"
        )
        logger.error(f"   Expected: {expected_gene_count} genes")
        logger.error(f"   Actual: {actual_gene_count} genes")
        logger.error(f"   Difference: {actual_gene_count - expected_gene_count}")
        logger.error("🚨 REFUSING TO SAVE CORRUPTED DATA - will regenerate alignment")
        return False

    # Use atomic write by writing to temp file first
    temp_path = result_path + ".tmp"

    for attempt in range(max_retries):
        try:
            # Write to temporary file
            result_df.to_csv(temp_path)

            # Verify the temporary file was written correctly
            verification_df = pd.read_csv(temp_path, index_col=0)
            if verification_df.empty:
                raise ValueError("Temporary file is empty after write")

            # Double-check gene count after write
            if len(verification_df) != expected_gene_count:
                raise ValueError(
                    f"Gene count mismatch after write: {len(verification_df)} != {expected_gene_count}"
                )

            # Atomic rename
            if os.path.exists(result_path):
                backup_path = result_path + ".bak"
                shutil.move(result_path, backup_path)

            shutil.move(temp_path, result_path)

            # Clean up backup if successful
            if os.path.exists(result_path + ".bak"):
                os.remove(result_path + ".bak")

            logger.debug(f"Successfully saved result to {result_path}")
            return True

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} to save {result_path} failed: {e}")

            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                logger.error(
                    f"Failed to save {result_path} after {max_retries} attempts"
                )

    return False


def cleanup_temp_files(srr, input_dir, keep_fastq=False):
    """
    Clean up temporary files for an SRR, optionally keeping FASTQ files.
    """
    logger.debug(f"Cleaning up temporary files for {srr}")

    # Lock files
    lock_path = os.path.join(input_dir, f"{srr}.lock")
    force_remove_file(lock_path)

    # FASTQ files (if not keeping them)
    if not keep_fastq:
        fastq_patterns = [f"{srr}.fastq", f"{srr}_1.fastq", f"{srr}_2.fastq"]

        for pattern in fastq_patterns:
            fastq_path = os.path.join(input_dir, pattern)
            if os.path.exists(fastq_path):
                force_remove_file(fastq_path)
                logger.debug(f"Removed FASTQ file: {fastq_path}")

    # SRA files
    sra_path = os.path.join(input_dir, f"{srr}.sra")
    if os.path.exists(sra_path):
        force_remove_file(sra_path)
        logger.debug(f"Removed SRA file: {sra_path}")


def create_intermediate_backup(output_dir, study_id=None):
    """
    Create a backup of intermediate results using simple file copying.
    """
    if not study_id:
        logger.warning("No study_id provided for backup")
        return False

    backup_dir = os.path.join(output_dir, f"{study_id}_backup")
    os.makedirs(backup_dir, exist_ok=True)

    try:
        # Copy all CSV result files
        result_files = [f for f in os.listdir(output_dir) if f.endswith("_result.csv")]

        for result_file in result_files:
            src_path = os.path.join(output_dir, result_file)
            dst_path = os.path.join(backup_dir, result_file)
            shutil.copy2(src_path, dst_path)

        logger.info(
            f"Created backup of {len(result_files)} result files in {backup_dir}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return False


def check_disk_space(directory, min_gb=10):
    """Check if there's enough disk space available."""
    try:
        total, used, free = shutil.disk_usage(directory)
        free_gb = free // (1024**3)

        logger.info(f"Disk space in {directory}: {free_gb} GB free")

        if free_gb < min_gb:
            logger.warning(
                f"Low disk space: only {free_gb} GB available in {directory}"
            )
            return False
        return True
    except Exception as e:
        logger.error(f"Could not check disk space for {directory}: {str(e)}")
        return True  # Assume it's OK if we can't check
