#!/usr/bin/env python3
"""
Process SRR accession numbers using archs4py by:
1. Doanloading each SRR in series
2. Aligning each SRR as it becomes available
3. Combining all results at the end

Note: Cannot use multiprocessing for downloading and alignment,
so downloads and alignments are done in series, but they can
happen simultaneously.
"""

import concurrent.futures
import logging
import os
import shutil
import time

import archs4py as a4
import archs4py.align as a4_align
import certifi
import pandas as pd

# Try to use certifi's built-in certificate path
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("process_srr.log")],
)
logger = logging.getLogger(__name__)


def wait_for_file(file_path, timeout=600, check_interval=30, post_detection_wait=5):
    """Wait for a file to become fully available and not busy."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            try:
                # Try to open the file to see if it's ready
                with open(file_path, "rb") as f:
                    f.read(1024)

                # File exists and is readable, but let's wait a moment to ensure
                # it's fully released by any writing processes
                time.sleep(post_detection_wait)

                # Try one more time to make absolutely sure
                with open(file_path, "rb") as f:
                    f.read(1)

                # If we get here, the file is ready
                logger.info(f"File {file_path} verified as ready and accessible")
                return True

            except OSError as e:
                logger.info(
                    f"File {file_path} exists but is not ready yet: "
                    f"{str(e)}. Waiting..."
                )
        else:
            logger.info(f"File {file_path} does not exist yet. Waiting...")

        time.sleep(check_interval)

    logger.error(f"Timeout waiting for file {file_path}")
    return False


def download_srr(srr, input_dir, max_retries=3):
    """Download a single SRR with retries and verification."""
    fastq_path = os.path.join(input_dir, f"{srr}.fastq")
    lock_path = os.path.join(input_dir, f"{srr}.lock")

    # Skip if file already exists and is readable
    if os.path.exists(fastq_path):
        try:
            with open(fastq_path, "rb") as f:
                f.read(1024)
            logger.info(
                f"FASTQ for {srr} already exists and is readable, skipping download."
            )
            return srr, True
        except OSError:
            logger.warning(
                f"FASTQ for {srr} exists but is not readable. "
                f"Will attempt to redownload."
            )
            # Try to remove the unreadable file
            try:
                os.remove(fastq_path)
                logger.info(f"Removed unreadable file {fastq_path}")
            except:  # noqa: E722
                pass

    # Create lock file
    try:
        with open(lock_path, "w") as f:
            f.write(f"Download started at {time.ctime()}")
    except Exception as e:
        logger.warning(f"Failed to create lock file for {srr}: {str(e)}")

    try:
        for retry in range(max_retries):
            try:
                # Load SRR data with a timeout for the whole operation
                logger.info(f"Downloading {srr} (attempt {retry+1}/{max_retries})...")

                # Start download with safety timeout of 2 hours
                start_time = time.time()
                max_download_time = 7200  # 2 hours

                a4_align.load([srr], input_dir)

                # Check if download took too long
                if time.time() - start_time > max_download_time:
                    logger.error(
                        f"Download of {srr} timed out after {max_download_time} seconds"
                    )
                    continue

                # Wait for file to become available
                if wait_for_file(
                    fastq_path, timeout=1800, check_interval=60, post_detection_wait=10
                ):
                    logger.info(f"Download of {srr} completed and verified.")
                    return srr, True
            except Exception as e:
                logger.error(f"Error downloading {srr} (attempt {retry+1}): {str(e)}")
                time.sleep(15)  # Longer wait before retry

        logger.error(f"Failed to download {srr} after {max_retries} attempts")
        return srr, False
    finally:
        # Always remove lock file, even if download fails
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception as e:
            logger.warning(f"Failed to remove lock file for {srr}: {str(e)}")


def align_srr(
    srr,
    input_dir,
    output_dir,
    genome="human",
    return_type="gene",
    identifier="symbol",
    max_retries=3,
    cleanup_after=False,
):
    """Align a single SRR with retries and optional cleanup."""
    fastq_path = os.path.join(input_dir, f"{srr}.fastq")
    result_path = os.path.join(output_dir, f"{srr}_result.csv")

    # Skip if result already exists and is readable
    if os.path.exists(result_path):
        try:
            result_df = pd.read_csv(result_path, index_col=0)
            logger.info(
                f"Result for {srr} already exists and is readable, skipping alignment."
            )

            # Cleanup if requested
            if cleanup_after and os.path.exists(fastq_path):
                try:
                    os.remove(fastq_path)
                    logger.info(
                        f"Deleted FASTQ file {fastq_path} after successful alignment"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to delete FASTQ file {fastq_path}: {str(e)}"
                    )

            return result_df
        except Exception as e:
            logger.warning(
                f"Error reading existing result for {srr}: {str(e)}. Will realign."
            )

    # Verify the fastq file exists and is ready
    if not wait_for_file(fastq_path):
        logger.error(f"FASTQ file for {srr} not found or not accessible, cannot align.")
        return None

    for retry in range(max_retries):
        try:
            # Align the fastq
            logger.info(f"Aligning {srr} (attempt {retry+1}/{max_retries})...")
            result = a4_align.fastq(
                genome, fastq_path, return_type=return_type, identifier=identifier
            )

            # Handle both Series and DataFrame outputs
            if isinstance(result, pd.Series):
                # Convert Series to DataFrame with SRR as column name
                logger.info(f"Result for {srr} is a Series, converting to DataFrame")
                result = pd.DataFrame(result)
                result.columns = [srr]
            elif isinstance(result, pd.DataFrame):
                # Rename the column to the SRR number
                result.columns = [srr]
            else:
                logger.error(
                    f"Error: Alignment for {srr} "
                    f"returned unexpected type: {type(result)}"
                )
                continue  # Try again

            # Save individual result with SRR identifier
            result_with_id = result.copy()
            result_with_id["srr_id"] = srr

            # Try to save with retries
            for save_retry in range(3):
                try:
                    result_with_id.to_csv(result_path)
                    logger.info(f"Saved result for {srr} to {result_path}")
                    break
                except Exception as e:
                    logger.warning(
                        f"Error saving result for {srr} "
                        f"(attempt {save_retry+1}): {str(e)}"
                    )
                    time.sleep(2)

            # Cleanup if requested
            if cleanup_after and os.path.exists(fastq_path):
                try:
                    os.remove(fastq_path)
                    logger.info(
                        f"Deleted FASTQ file {fastq_path} after successful alignment"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to delete FASTQ file {fastq_path}: {str(e)}"
                    )

            return result
        except Exception as e:
            logger.error(f"Error aligning {srr} (attempt {retry+1}): {str(e)}")
            time.sleep(5)  # Wait before retry

    logger.error(f"Failed to align {srr} after {max_retries} attempts")
    return None


def process_srrs_pipeline(
    srr_list,
    input_dir,
    output_dir,
    genome="human",
    return_type="gene",
    identifier="symbol",
    cleanup="end",
):
    """
    Process SRRs with controlled parallelism:
    - One download at a time
    - One alignment at a time
    - Download and alignment can happen simultaneously

    cleanup options:
    - "none": Keep all FASTQ files
    - "immediate": Delete FASTQ files immediately after alignment
    - "end": Delete all FASTQ files at the end of processing
    """
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    failed_downloads = []
    failed_alignments = []

    # Check for already downloaded files first
    ready_for_alignment = []
    to_download = []

    for srr in srr_list:
        fastq_path = os.path.join(input_dir, f"{srr}.fastq")
        if os.path.exists(fastq_path) and os.path.getsize(fastq_path) > 0:
            try:
                with open(fastq_path, "rb") as f:
                    f.read(1)
                ready_for_alignment.append(srr)
            except OSError:
                to_download.append(srr)
        else:
            to_download.append(srr)

    logger.info(
        f"{len(ready_for_alignment)} SRRs already downloaded, "
        f"{len(to_download)} need downloading"
    )

    # Create two single-worker executors, archs4py.align.load and archs4py.align.fastq
    # are not thread-safe. One load and one fastq at a time.
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=1) as download_executor,
        concurrent.futures.ThreadPoolExecutor(max_workers=1) as align_executor,
    ):
        # Track active futures
        active_download_future = None
        active_align_future = None
        current_download_srr = None
        current_align_srr = None

        # First process already downloaded files
        if ready_for_alignment:
            srr = ready_for_alignment.pop(0)
            active_align_future = align_executor.submit(
                align_srr,
                srr,
                input_dir,
                output_dir,
                genome,
                return_type,
                identifier,
                max_retries=3,
                cleanup_after=(cleanup == "immediate"),
            )
            current_align_srr = srr
            logger.info(f"Started alignment of existing file {srr}")

        # Start the first download if any
        if to_download:
            srr = to_download.pop(0)
            active_download_future = download_executor.submit(
                download_srr, srr, input_dir
            )
            current_download_srr = srr
            logger.info(f"Started download of {srr}")

        # Process until all downloads and alignments are complete
        while (
            active_download_future is not None
            or active_align_future is not None
            or ready_for_alignment
            or to_download
        ):
            # Check if download is complete
            if active_download_future is not None and active_download_future.done():
                try:
                    srr, success = active_download_future.result()
                    if success:
                        logger.info(f"Download of {srr} completed")
                        ready_for_alignment.append(srr)
                    else:
                        logger.error(f"Download failed for {srr}")
                        failed_downloads.append(srr)
                except Exception as e:
                    logger.error(
                        f"Error in download of {current_download_srr}: {str(e)}"
                    )
                    failed_downloads.append(current_download_srr)

                # Start next download
                active_download_future = None
                current_download_srr = None
                if to_download:
                    srr = to_download.pop(0)
                    active_download_future = download_executor.submit(
                        download_srr, srr, input_dir
                    )
                    current_download_srr = srr
                    logger.info(f"Started download of {srr}")

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

                # Start next alignment
                active_align_future = None
                current_align_srr = None
                if ready_for_alignment:
                    srr = ready_for_alignment.pop(0)
                    active_align_future = align_executor.submit(
                        align_srr,
                        srr,
                        input_dir,
                        output_dir,
                        genome,
                        return_type,
                        identifier,
                        max_retries=3,
                        cleanup_after=(cleanup == "immediate"),
                    )
                    current_align_srr = srr
                    logger.info(f"Started alignment of {srr}")

            # If no alignment is running but we have files ready, start one
            if (
                active_align_future is None
                and ready_for_alignment
                and current_align_srr is None
            ):
                srr = ready_for_alignment.pop(0)
                active_align_future = align_executor.submit(
                    align_srr,
                    srr,
                    input_dir,
                    output_dir,
                    genome,
                    return_type,
                    identifier,
                    max_retries=3,
                    cleanup_after=(cleanup == "immediate"),
                )
                current_align_srr = srr
                logger.info(f"Started alignment of {srr}")

            # Sleep briefly to avoid busy waiting
            time.sleep(1)

    # Try to combine all results
    if all_results:
        try:
            logger.info(
                f"Combining results from {len(all_results)} successful alignments"
            )

            # Ensure all results are DataFrames with proper columns before concatenation
            normalized_results = []
            for result in all_results:
                if isinstance(result, pd.Series):
                    # Convert any Series to DataFrame
                    srr_id = (
                        result.name
                        if hasattr(result, "name") and result.name is not None
                        else "unknown"
                    )
                    normalized_results.append(
                        pd.DataFrame(result, columns=pd.Index([srr_id]))
                    )
                elif isinstance(result, pd.DataFrame):
                    normalized_results.append(result)
                else:
                    logger.warning(f"Skipping unexpected result type: {type(result)}")

            combined_df = pd.concat(normalized_results, axis=1)
            combined_path = os.path.join(output_dir, "combined_results.csv")
            combined_df.to_csv(combined_path)
            logger.info(f"Saved combined results to {combined_path}")

            # Also save a summary of successes and failures
            summary = {
                "total_srrs": len(srr_list),
                "successful_downloads": len(srr_list) - len(failed_downloads),
                "failed_downloads": failed_downloads,
                "successful_alignments": len(all_results),
                "failed_alignments": failed_alignments,
            }

            with open(os.path.join(output_dir, "processing_summary.txt"), "w") as f:
                for key, value in summary.items():
                    if isinstance(value, list):
                        f.write(f"{key}: {len(value)}\n")
                        f.write(f"{key}_list: {', '.join(value)}\n")
                    else:
                        f.write(f"{key}: {value}\n")

            # Cleanup if requested
            if cleanup == "end":
                logger.info("Cleaning up FASTQ files...")
                cleanup_count = 0
                for srr in srr_list:
                    fastq_path = os.path.join(input_dir, f"{srr}.fastq")
                    if os.path.exists(fastq_path):
                        try:
                            os.remove(fastq_path)
                            cleanup_count += 1
                        except Exception as e:
                            logger.warning(
                                f"Failed to delete FASTQ file {fastq_path}: {str(e)}"
                            )
                logger.info(f"Deleted {cleanup_count} FASTQ files during final cleanup")

            return combined_df
        except Exception as e:
            logger.error(f"Error combining results: {str(e)}")
            return None
    else:
        logger.error("No valid alignment results to combine.")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process SRR accessions in series")
    parser.add_argument(
        "--srr_file",
        required=True,
        help="Path to file with SRR accessions (one per line)",
    )
    parser.add_argument(
        "--input_dir", default="data/input", help="Directory for downloaded FASTQ files"
    )
    parser.add_argument(
        "--output_dir", default="data/output", help="Directory for output count files"
    )
    parser.add_argument("--genome", default="human", help="Genome to align to")
    parser.add_argument(
        "--return_type", default="gene", help="Return type for alignment"
    )
    parser.add_argument(
        "--identifier", default="symbol", help="Identifier type for alignment"
    )
    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="Run in batch mode with periodic saves",
    )
    parser.add_argument(
        "--cleanup",
        choices=["none", "immediate", "end"],
        default="end",
        help="Cleanup strategy for FASTQ files",
    )
    parser.add_argument(
        "--reset-archs4",
        action="store_true",
        help="Reset ARCHS4 data files before processing",
    )

    args = parser.parse_args()

    # Reset ARCHS4 if requested. Sometimes required to fix corrupted files.
    if args.reset_archs4:
        archs4_path = os.path.dirname(a4.__file__)
        xalign_path = os.path.join(os.path.dirname(archs4_path), "xalign")
        data_path = os.path.join(xalign_path, "data")

        logger.info(f"Resetting ARCHS4 data at {data_path}")

        # Clean directories that might contain corrupted files
        for dir_name in ["outkallisto", "kallisto", "fasterq", "index"]:
            dir_path = os.path.join(data_path, dir_name)
            if os.path.exists(dir_path):
                logger.info(f"Removing directory: {dir_path}")
                shutil.rmtree(dir_path)

        # Remove the problematic FASTA file
        fasta_file = os.path.join(data_path, "homo_sapiens.107.fastq.gz")
        if os.path.exists(fasta_file):
            logger.info(f"Removing file: {fasta_file}")
            os.remove(fasta_file)

        logger.info("ARCHS4 data reset completed")

    # Read SRR accessions
    with open(args.srr_file) as f:
        srr_list = [line.strip() for line in f if line.strip()]

    logger.info(f"Read {len(srr_list)} SRR accessions from {args.srr_file}")

    # Process them
    process_srrs_pipeline(
        srr_list,
        args.input_dir,
        args.output_dir,
        args.genome,
        args.return_type,
        args.identifier,
    )


if __name__ == "__main__":
    main()
