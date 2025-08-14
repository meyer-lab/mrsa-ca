#!/usr/bin/env python3
"""
FASTQ file utilities for the RNA-seq processing pipeline.
"""

import logging
import os
import time

logger = logging.getLogger(__name__)


def detect_fastq_type(srr, input_dir):
    """
    Detect whether FASTQ files are single-end or paired-end.
    Returns 'single', 'paired', or 'missing'.
    """
    single_path = os.path.join(input_dir, f"{srr}.fastq")
    paired_path1 = os.path.join(input_dir, f"{srr}_1.fastq")
    paired_path2 = os.path.join(input_dir, f"{srr}_2.fastq")

    # Check for single-end file
    if os.path.exists(single_path):
        return "single"

    # Check for paired-end files
    elif os.path.exists(paired_path1) and os.path.exists(paired_path2):
        return "paired"

    # Check if only one paired file exists (incomplete download)
    elif os.path.exists(paired_path1) or os.path.exists(paired_path2):
        logger.warning(f"Incomplete paired-end files for {srr}")
        return "incomplete"

    # No FASTQ files found
    else:
        return "missing"


def wait_for_paired_end_files(srr, input_dir, timeout=1800, check_interval=60):
    """
    Wait for both paired-end files to become available.
    Used when downloads are happening in parallel.
    """
    paired_path1 = os.path.join(input_dir, f"{srr}_1.fastq")
    paired_path2 = os.path.join(input_dir, f"{srr}_2.fastq")

    start_time = time.time()

    while time.time() - start_time < timeout:
        if os.path.exists(paired_path1) and os.path.exists(paired_path2):
            # Both files exist, check if they're complete
            try:
                # Quick size check - both files should have reasonable size
                size1 = os.path.getsize(paired_path1)
                size2 = os.path.getsize(paired_path2)

                if size1 > 1000 and size2 > 1000:  # At least 1KB each
                    logger.info(
                        f"Paired-end files ready for {srr}: "
                        f"{size1 / 1024 / 1024:.1f}MB + {size2 / 1024 / 1024:.1f}MB"
                    )
                    return True
                else:
                    logger.debug(
                        f"Paired-end files exist but seem small for {srr}, waiting..."
                    )

            except Exception as e:
                logger.debug(f"Error checking paired-end file sizes for {srr}: {e}")

        time.sleep(check_interval)

    logger.error(f"Timeout waiting for paired-end files for {srr}")
    return False


def get_fastq_paths(srr, input_dir):
    """
    Get the appropriate FASTQ file paths for an SRR.
    Returns (read_type, file_paths) where file_paths is a list.
    """
    read_type = detect_fastq_type(srr, input_dir)

    if read_type == "single":
        single_path = os.path.join(input_dir, f"{srr}.fastq")
        return "single", [single_path]

    elif read_type == "paired":
        paired_path1 = os.path.join(input_dir, f"{srr}_1.fastq")
        paired_path2 = os.path.join(input_dir, f"{srr}_2.fastq")
        return "paired", [paired_path1, paired_path2]

    else:
        return read_type, []


def validate_fastq_files(file_paths):
    """
    Validate that FASTQ files exist and are readable.
    """
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.error(f"FASTQ file not found: {file_path}")
            return False

        try:
            # Try to read first few lines to validate format
            with open(file_path) as f:
                first_line = f.readline().strip()
                if not first_line.startswith("@"):
                    logger.error(f"Invalid FASTQ format in {file_path}: {first_line}")
                    return False
        except Exception as e:
            logger.error(f"Cannot read FASTQ file {file_path}: {e}")
            return False

    return True
