#!/usr/bin/env python3
"""
Sample Mapper - A tool for mapping GSM accessions to SRR accessions

This script takes a GSE accession number and creates mapping files integrated
with the RNA-seq processing pipeline directory structure.

Example usage:
  python sample_mapper.py GSE123456
  python sample_mapper.py GSE123456 --base-dir data
  python sample_mapper.py --file gse_list.txt --characteristics

Output:
  - Creates mapping JSON files in data/accessions/: GSE123456_mapping.json
  - Creates metadata JSON files in data/results/GSE123456/:
    GSE123456_characteristics.json

  Mapping file structure:
  {
      "GSM1234567": ["SRR1234567", "SRR1234568"],
      "GSM1234568": ["SRR1234569"],
      ...
  }

Integration with pipeline:
  - Mapping files in data/accessions/ can be used with
    process_rnaseq.py --sample_mapping
  - Metadata files are organized by study in data/results/{GSE}/ for easy
    pairing with alignment results
  - Directory structure matches the pipeline's expectations for checkpoint
    and result organization
"""

import argparse
import json
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET

import GEOparse
import requests

#############################################
# Configuration
#############################################

# NCBI E-utilities base URL
base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# NCBI API key for E-utilities
ncbi_api_key = "7f7ac246396cc1bd1f47c090de5ddebeb709"

# Rate limiting: Conservative approach with ~3 requests per second
# NCBI guidelines suggest up to 10 requests/second with API key, but
# connection errors indicate we need to be more conservative
RATE_LIMIT_DELAY = 0.35  # ~3 requests per second
# Take a break after this many GSM mappings (within a single GSE)
GSM_BATCH_SIZE = 250
BATCH_BREAK_TIME = 120  # Break time in seconds (2 minutes)
GSE_BREAK_TIME = 360  # Break time between GSE processing (6 minutes)

#############################################
# API Functions
#############################################


def make_api_request(url, retry_delay=0.5, max_retries=5):
    """Make an API request with retry logic for handling rate limits and
    connection errors.

    Parameters
    ----------
    url : str
        API url for the request
    retry_delay : float, optional
        Base seconds to wait between retries, by default 0.5
    max_retries : int, optional
        Maximum number of retry attempts to fetch, by default 5

    Returns
    -------
    requests.Response or None
        Response object from the requests library if successful, else None
    """
    for retry in range(max_retries):
        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                wait_time = retry_delay * (retry + 1)
                print(f"Rate limit hit, retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                # Exponential backoff with higher cap
                retry_delay = min(retry_delay * 2, 15)
            else:
                print(f"Error: HTTP {response.status_code}")
                if response.content:
                    print(f"Error details: {response.content[:200]}...")
                break

        except requests.exceptions.ConnectionError:
            wait_time = retry_delay * (retry + 1)
            print(
                f"Connection error on attempt {retry + 1}/{max_retries}, "
                f"retrying in {wait_time:.1f} seconds..."
            )
            if retry < max_retries - 1:  # Don't sleep on the last attempt
                time.sleep(wait_time)
            retry_delay = min(retry_delay * 2, 15)  # Exponential backoff

        except requests.exceptions.Timeout:
            wait_time = retry_delay * (retry + 1)
            print(
                f"Timeout error on attempt {retry + 1}/{max_retries}, "
                f"retrying in {wait_time:.1f} seconds..."
            )
            if retry < max_retries - 1:  # Don't sleep on the last attempt
                time.sleep(wait_time)
            retry_delay = min(retry_delay * 2, 15)  # Exponential backoff

        except Exception as e:
            print(f"Unexpected error: {e}")
            break

    return None


def get_gsm_accessions_from_gse(gse_accession):
    """Get all GSM accessions for a given GSE accession.

    Parameters
    ----------
    gse_accession : str
        The GSE accession number (e.g., GSE123456)

    Returns
    -------
    list[str]
        List of GSM accession numbers
    """
    print(f"Fetching GSM accessions for {gse_accession}...")

    # Step 1: Search for the GSE accession to get the UID
    search_url = (
        f"{base_url}esearch.fcgi?db=gds&term={gse_accession}[Accession]"
        f"&api_key={ncbi_api_key}"
    )

    search_response = make_api_request(search_url)
    if not search_response:
        print(f"Failed to find UID for accession {gse_accession}")
        return []

    # Add delay between requests to be more conservative
    time.sleep(RATE_LIMIT_DELAY)  # ~3 requests per second

    try:
        search_root = ET.fromstring(search_response.content)
        id_elems = search_root.findall(".//Id")

        if not id_elems:
            print(f"No UID found for {gse_accession}")
            return []

        uid = id_elems[0].text
        print(f"Found UID {uid} for accession {gse_accession}")

        # Step 2: Use the UID to fetch complete details including samples
        fetch_url = f"{base_url}esummary.fcgi?db=gds&id={uid}&api_key={ncbi_api_key}"

        fetch_response = make_api_request(fetch_url)
        if not fetch_response:
            return []

        # Add delay between requests to be more conservative
        time.sleep(RATE_LIMIT_DELAY)  # ~3 requests per second

        # Parse the XML response
        root = ET.fromstring(fetch_response.content)
        doc = root.find(".//DocSum")

        if doc is None:
            print(f"No data found for {gse_accession}")
            return []

        # Extract GSM accessions
        gsm_accessions = []
        samples_item = doc.find("./Item[@Name='Samples']")

        if samples_item is None:
            print(f"No samples found for {gse_accession}")
            return []

        # Extract each sample accession
        for sample in samples_item.findall("./Item[@Name='Sample']"):
            accession_item = sample.find("./Item[@Name='Accession']")
            if accession_item is not None:
                gsm_accessions.append(accession_item.text)

        print(f"Found {len(gsm_accessions)} GSM accessions for {gse_accession}")
        return gsm_accessions

    except ET.ParseError as e:
        print(f"Error parsing XML for accession {gse_accession}: {e}")
        return []


def gsm_to_srr(gsm_accession, retry_delay=RATE_LIMIT_DELAY):
    """Convert a GSM accession to SRR accessions.

    Parameters
    ----------
    gsm_accession : str
        The GSM accession number
    retry_delay : float, optional
        Delay between API requests in seconds, by default RATE_LIMIT_DELAY
        (~3 requests/second)

    Returns
    -------
    list[str]
        List of SRR accession numbers
    """
    time.sleep(retry_delay)  # Respect rate limits - conservative approach

    # Search for the SRA entry linked to this GSM
    search_url = (
        f"{base_url}esearch.fcgi?db=sra&term={gsm_accession}&api_key={ncbi_api_key}"
    )

    response = make_api_request(search_url, retry_delay)
    if not response:
        return []

    try:
        root = ET.fromstring(response.content)
        count = int(root.find("Count").text) if root.find("Count") is not None else 0

        if count == 0:
            print(f"No SRA entries found for {gsm_accession}")
            return []

        # Get the SRA UIDs
        sra_uids = [id_elem.text for id_elem in root.findall(".//Id")]

        if not sra_uids:
            return []

        # Use efetch with runinfo format
        time.sleep(retry_delay)  # Conservative delay before second request
        fetch_url = (
            f"{base_url}efetch.fcgi?db=sra&id={','.join(sra_uids)}"
            f"&rettype=runinfo&api_key={ncbi_api_key}"
        )

        fetch_response = make_api_request(fetch_url, retry_delay)
        if not fetch_response:
            return []

        # Parse the response - try XML first, then CSV fallback
        try:
            runinfo_root = ET.fromstring(fetch_response.text)
            srr_accessions = []

            for row in runinfo_root.findall(".//Row"):
                run_elem = row.find("Run")
                if run_elem is not None and run_elem.text:
                    srr_accessions.append(run_elem.text)

            return srr_accessions

        except ET.ParseError:
            # Fall back to CSV parsing
            lines = fetch_response.text.strip().split("\n")
            if len(lines) < 2:  # Need at least header + 1 data row
                return []

            # Parse CSV for Run column
            header = lines[0].split(",")

            if "Run" not in header:
                return []

            run_index = header.index("Run")
            srr_accessions = []

            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) > run_index and parts[run_index].strip():
                    srr_accessions.append(parts[run_index].strip())

            return srr_accessions

    except Exception as e:
        print(f"Error processing SRA data for {gsm_accession}: {str(e)}")
        return []


def fetch_gsm_metadata(gsm_accession, retry_delay=RATE_LIMIT_DELAY):
    """Fetch detailed metadata for a GSM sample using GEOparse library.

    Parameters
    ----------
    gsm_accession : str
        The GSM accession number
    retry_delay : float, optional
        Delay between API requests, by default RATE_LIMIT_DELAY (~3 requests/second)

    Returns
    -------
    dict
        Dictionary containing sample metadata including characteristics
    """
    print(f"Fetching metadata for {gsm_accession} with GEOparse...")

    # Add delay to respect rate limits
    time.sleep(retry_delay)

    try:
        # Create a temporary directory for GEOparse files
        with tempfile.TemporaryDirectory() as tmp_dir:
            # GEOparse will download the GSM data to the temp directory
            gsm = GEOparse.get_GEO(geo=gsm_accession, silent=True, destdir=tmp_dir)

            metadata = {
                "characteristics": {},
                "platform": gsm.metadata.get("platform_id", [""])[0],
                "instrument": gsm.metadata.get("instrument_model", [""])[0],
                "library_strategy": gsm.metadata.get("library_strategy", [""])[0],
                "submission_date": gsm.metadata.get("submission_date", [""])[0],
                "channel_count": gsm.metadata.get("channel_count", [""])[0],
            }

            # Extract characteristics into a flat dictionary
            if "characteristics_ch1" in gsm.metadata:
                for char_entry in gsm.metadata["characteristics_ch1"]:
                    # Format is typically "key: value"
                    if ":" in char_entry:
                        key, value = char_entry.split(":", 1)
                        safe_key = (
                            key.strip().replace(" ", "_").replace("-", "_").lower()
                        )
                        metadata["characteristics"][safe_key] = value.strip()

            # When this block ends, the temporary directory and its contents are deleted

        return metadata

    except Exception as e:
        print(f"Error fetching metadata for {gsm_accession} with GEOparse: {str(e)}")
        # Fall back to empty characteristics if GEOparse fails
        return {"characteristics": {}}


def create_gsm_to_srr_mapping(
    gse_accession, base_dir="data", progress=True, fetch_characteristics=False
):
    """Create a JSON mapping file for GSM to SRR accessions.

    Parameters
    ----------
    gse_accession : str
        The GSE accession number
    base_dir : str, optional
        Base directory for data files, by default "data"
        Mapping files will be saved to {base_dir}/accessions/
        Metadata files will be saved to {base_dir}/results/{gse_accession}/
    progress : bool, optional
        Whether to show progress updates, by default True
    fetch_characteristics : bool, optional
        Whether to also fetch sample characteristics, by default False

    Returns
    -------
    dict
        The mapping dictionary
    """
    # Add initial delay to space out requests when processing multiple GSE accessions
    time.sleep(0.5)

    # Check if files already exist
    accessions_dir = os.path.join(base_dir, "accessions")
    results_dir = os.path.join(base_dir, "results", gse_accession)
    mapping_file = os.path.join(accessions_dir, f"{gse_accession}_mapping.json")
    characteristics_file = os.path.join(
        results_dir, f"{gse_accession}_characteristics.json"
    )

    mapping_exists = os.path.exists(mapping_file)
    characteristics_exist = os.path.exists(characteristics_file)

    # Determine what needs to be processed
    skip_mapping = mapping_exists
    skip_characteristics = not fetch_characteristics or characteristics_exist

    if skip_mapping and skip_characteristics:
        print(f"Files for {gse_accession} already exist. Skipping processing.")
        if mapping_exists and progress:
            print(f"  Mapping file exists: {mapping_file}")
        if characteristics_exist and fetch_characteristics and progress:
            print(f"  Characteristics file exists: {characteristics_file}")

        # Load and return existing mapping
        if mapping_exists:
            try:
                with open(mapping_file) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading existing mapping file {mapping_file}: {e}")
                print("Will regenerate the file...")
                skip_mapping = False
        else:
            return {}

    if skip_mapping and not skip_characteristics:
        print(
            f"Mapping file for {gse_accession} already exists. "
            f"Only fetching characteristics."
        )
        if progress:
            print(f"  Existing mapping file: {mapping_file}")

        # Load existing mapping and just fetch characteristics
        try:
            with open(mapping_file) as f:
                mapping = json.load(f)

            # Get GSM accessions from existing mapping
            gsm_accessions = list(mapping.keys())

            if gsm_accessions and fetch_characteristics:
                print("\n" + "=" * 50)
                create_gsm_characteristics_mapping(
                    gse_accession, gsm_accessions, base_dir, progress
                )

            return mapping
        except Exception as e:
            print(f"Error loading existing mapping file {mapping_file}: {e}")
            print("Will regenerate the file...")

    # Get all GSM accessions for this GSE
    gsm_accessions = get_gsm_accessions_from_gse(gse_accession)

    if not gsm_accessions:
        print(f"No GSM accessions found for {gse_accession}")
        return {}

    # Create the SRR mapping (only if not skipping)
    if not skip_mapping:
        mapping = {}
        total = len(gsm_accessions)

        print(f"Mapping {total} GSM accessions to SRR accessions...")
        if total > GSM_BATCH_SIZE:
            print(
                f"Note: Will take {BATCH_BREAK_TIME}-second breaks after "
                f"every {GSM_BATCH_SIZE} GSM mappings"
            )

        for i, gsm in enumerate(gsm_accessions, 1):
            if progress:
                print(f"Processing {i}/{total}: {gsm}")

            srr_accessions = gsm_to_srr(gsm)
            mapping[gsm] = srr_accessions

            if progress and srr_accessions:
                print(f"  Found {len(srr_accessions)} SRR accessions: {srr_accessions}")
            elif progress:
                print("  No SRR accessions found")

            # Take a break after every GSM_BATCH_SIZE mappings
            if i % GSM_BATCH_SIZE == 0 and i < total:
                print(
                    f"\n🛑 Taking a {BATCH_BREAK_TIME}-second break "
                    f"after {i} GSM mappings..."
                )
                print(f"   Progress: {i}/{total} ({i / total * 100:.1f}%) complete")
                remaining_time = (
                    (total - i) * RATE_LIMIT_DELAY
                    + (total - i) // GSM_BATCH_SIZE * BATCH_BREAK_TIME
                ) / 60
                print(f"   Estimated remaining time: {remaining_time:.1f} minutes")
                for countdown in range(BATCH_BREAK_TIME, 0, -10):
                    print(f"   Resuming in {countdown} seconds...", end="\r")
                    time.sleep(10)
                print("   Resuming processing...                    ")
                print("🚀 Continuing with GSM mapping...\n")

        # Save SRR mapping to JSON file in data/accessions directory
        os.makedirs(accessions_dir, exist_ok=True)

        with open(mapping_file, "w") as f:
            json.dump(mapping, f, indent=2)

        print(f"\nMapping saved to: {mapping_file}")
        print(f"Total GSM accessions: {len(mapping)}")
        print(
            f"GSM accessions with SRR data: "
            f"{sum(1 for srrs in mapping.values() if srrs)}"
        )
    else:
        # Load existing mapping
        with open(mapping_file) as f:
            mapping = json.load(f)

    # Fetch characteristics if requested and not skipping
    if fetch_characteristics and not skip_characteristics:
        print("\n" + "=" * 50)
        create_gsm_characteristics_mapping(
            gse_accession, gsm_accessions, base_dir, progress
        )

    return mapping


def create_gsm_characteristics_mapping(
    gse_accession, gsm_accessions, base_dir="data", progress=True
):
    """Create a JSON file with GSM to characteristics mapping.

    Parameters
    ----------
    gse_accession : str
        The GSE accession number
    gsm_accessions : list[str]
        List of GSM accession numbers
    base_dir : str, optional
        Base directory for data files, by default "data"
        Metadata files will be saved to {base_dir}/results/{gse_accession}/
    progress : bool, optional
        Whether to show progress updates, by default True

    Returns
    -------
    dict
        The characteristics mapping dictionary
    """
    if not gsm_accessions:
        print("No GSM accessions provided for characteristics mapping")
        return {}

    # Check if characteristics file already exists
    results_dir = os.path.join(base_dir, "results", gse_accession)
    output_file = os.path.join(results_dir, f"{gse_accession}_characteristics.json")

    if os.path.exists(output_file):
        print(f"Characteristics file for {gse_accession} already exists: {output_file}")
        try:
            with open(output_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading existing characteristics file: {e}")
            print("Will regenerate the file...")

    # Create the mapping
    characteristics_mapping = {}
    total = len(gsm_accessions)

    print(f"Fetching characteristics for {total} GSM accessions...")
    if total > GSM_BATCH_SIZE:
        print(
            f"Note: Will take {BATCH_BREAK_TIME}-second breaks after "
            f"every {GSM_BATCH_SIZE} characteristics fetches"
        )

    for i, gsm in enumerate(gsm_accessions, 1):
        if progress:
            print(f"Processing characteristics {i}/{total}: {gsm}")

        metadata = fetch_gsm_metadata(gsm)
        characteristics_mapping[gsm] = metadata

        if progress and metadata.get("characteristics"):
            print(f"  Found {len(metadata['characteristics'])} characteristics")
        elif progress:
            print("  No characteristics found")

        # Take a break after every GSM_BATCH_SIZE characteristics fetches
        if i % GSM_BATCH_SIZE == 0 and i < total:
            print(
                f"\n🛑 Taking a {BATCH_BREAK_TIME}-second break "
                f"after {i} characteristics fetches..."
            )
            print(f"   Progress: {i}/{total} ({i / total * 100:.1f}%) complete")
            remaining_time = (
                (total - i) * RATE_LIMIT_DELAY
                + (total - i) // GSM_BATCH_SIZE * BATCH_BREAK_TIME
            ) / 60
            print(f"   Estimated remaining time: {remaining_time:.1f} minutes")
            for countdown in range(BATCH_BREAK_TIME, 0, -10):
                print(f"   Resuming in {countdown} seconds...", end="\r")
                time.sleep(10)
            print("   Resuming processing...                    ")
            print("🚀 Continuing with characteristics fetching...\n")

    # Save to JSON file in study-specific results directory
    os.makedirs(results_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(characteristics_mapping, f, indent=2)

    print(f"\nCharacteristics saved to: {output_file}")
    print(f"Total GSM accessions: {len(characteristics_mapping)}")
    gsm_with_characteristics = sum(
        1
        for metadata in characteristics_mapping.values()
        if metadata.get("characteristics")
    )
    print(f"GSM accessions with characteristics: {gsm_with_characteristics}")

    return characteristics_mapping


def read_gse_file(filepath):
    """Read GSE accessions from a text file.

    Parameters
    ----------
    filepath : str
        Path to the text file containing GSE accessions

    Returns
    -------
    list[str]
        List of GSE accession numbers
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return []

    accessions = []
    try:
        with open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments (lines starting with #)
                if not line or line.startswith("#"):
                    continue

                # Extract GSE accession (handle various formats)
                gse_match = re.search(r"GSE\d+", line.upper())
                if gse_match:
                    accessions.append(gse_match.group(0))
                else:
                    print(
                        f"Warning: Line {line_num} does not contain a "
                        f"valid GSE accession: '{line}'"
                    )

    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return []

    if accessions:
        print(f"Found {len(accessions)} GSE accessions in {filepath}")
    else:
        print(f"No valid GSE accessions found in {filepath}")

    return accessions


def process_multiple_gse(
    gse_list, base_dir="data", progress=True, fetch_characteristics=False
):
    """Process multiple GSE accessions from a list.

    Parameters
    ----------
    gse_list : list[str]
        List of GSE accession numbers
    base_dir : str, optional
        Base directory for data files, by default "data"
        Mapping files will be saved to {base_dir}/accessions/
        Metadata files will be saved to {base_dir}/results/{gse_accession}/
    progress : bool, optional
        Whether to show progress updates, by default True
    fetch_characteristics : bool, optional
        Whether to fetch sample characteristics, by default False

    Returns
    -------
    dict
        Dictionary with GSE accessions as keys and results as values
    """
    results = {}
    total_gse = len(gse_list)

    print(f"\nProcessing {total_gse} GSE accessions...")
    if total_gse > 1:
        print(f"Note: Will take {GSE_BREAK_TIME}-second breaks between each GSE")
    if fetch_characteristics:
        print("Sample characteristics will be fetched for each GSE.")

    for i, gse_accession in enumerate(gse_list, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing GSE {i}/{total_gse}: {gse_accession}")
        print("=" * 60)

        # Check if files already exist
        accessions_dir = os.path.join(base_dir, "accessions")
        results_dir = os.path.join(base_dir, "results", gse_accession)
        mapping_file = os.path.join(accessions_dir, f"{gse_accession}_mapping.json")
        characteristics_file = os.path.join(
            results_dir, f"{gse_accession}_characteristics.json"
        )

        mapping_exists = os.path.exists(mapping_file)
        characteristics_exist = os.path.exists(characteristics_file)

        # Determine what needs to be processed
        skip_mapping = mapping_exists
        skip_characteristics = not fetch_characteristics or characteristics_exist

        if skip_mapping and skip_characteristics:
            print(f"Files for {gse_accession} already exist. Skipping processing.")
            if mapping_exists and progress:
                print(f"  Mapping file exists: {mapping_file}")
            if characteristics_exist and fetch_characteristics and progress:
                print(f"  Characteristics file exists: {characteristics_file}")

            # Load existing mapping for statistics
            if mapping_exists:
                try:
                    with open(mapping_file) as f:
                        mapping = json.load(f)
                    results[gse_accession] = {
                        "status": "skipped",
                        "gsm_count": len(mapping),
                        "srr_count": sum(len(srrs) for srrs in mapping.values()),
                    }
                    print(f"⏭ {gse_accession} skipped (files exist)")
                    continue
                except Exception as e:
                    print(f"Error loading existing files for {gse_accession}: {e}")
                    print("Will regenerate the files...")
            else:
                results[gse_accession] = {
                    "status": "skipped",
                    "gsm_count": 0,
                    "srr_count": 0,
                }
                print(f"⏭ {gse_accession} skipped (files exist)")
                continue

        try:
            mapping = create_gsm_to_srr_mapping(
                gse_accession,
                base_dir=base_dir,
                progress=progress,
                fetch_characteristics=fetch_characteristics,
            )

            if mapping:
                results[gse_accession] = {
                    "status": "success",
                    "gsm_count": len(mapping),
                    "srr_count": sum(len(srrs) for srrs in mapping.values()),
                }
                print(f"✓ {gse_accession} completed successfully!")
            else:
                results[gse_accession] = {
                    "status": "no_data",
                    "gsm_count": 0,
                    "srr_count": 0,
                }
                print(f"⚠ {gse_accession} - No data found")

        except Exception as e:
            results[gse_accession] = {
                "status": "error",
                "error": str(e),
                "gsm_count": 0,
                "srr_count": 0,
            }
            print(f"✗ {gse_accession} failed: {e}")

        # Take a break between GSE processing (except after the last one)
        if i < total_gse:
            print(f"\n🛑 Taking a {GSE_BREAK_TIME}-second break before next GSE...")
            print(
                f"   Completed: {i}/{total_gse} GSE accessions "
                f"({i / total_gse * 100:.1f}%)"
            )
            completed_gsm = sum(r.get("gsm_count", 0) for r in results.values())
            completed_srr = sum(r.get("srr_count", 0) for r in results.values())
            print(
                f"   Total processed so far: {completed_gsm} GSM → "
                f"{completed_srr} SRR accessions"
            )
            for countdown in range(GSE_BREAK_TIME, 0, -5):
                print(f"   Next GSE in {countdown} seconds...", end="\r")
                time.sleep(5)
            print("   Continuing to next GSE...                    ")

    # Print summary
    print(f"\n{'=' * 60}")
    print("PROCESSING SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results.values() if r["status"] == "success")
    skipped = sum(1 for r in results.values() if r["status"] == "skipped")
    failed = sum(1 for r in results.values() if r["status"] == "error")
    no_data = sum(1 for r in results.values() if r["status"] == "no_data")
    total_gsm = sum(r["gsm_count"] for r in results.values())
    total_srr = sum(r["srr_count"] for r in results.values())

    print(f"Total GSE processed: {total_gse}")
    print(f"Successful: {successful}")
    print(f"Skipped (files exist): {skipped}")
    print(f"No data: {no_data}")
    print(f"Failed: {failed}")
    print(f"Total GSM accessions: {total_gsm}")
    print(f"Total SRR accessions: {total_srr}")

    if failed > 0:
        print("\nFailed GSE accessions:")
        for gse, result in results.items():
            if result["status"] == "error":
                print(f"  {gse}: {result['error']}")

    if skipped > 0:
        print(f"\nSkipped {skipped} GSE accessions with existing files.")
        if progress:
            print("Use --force flag (if implemented) to regenerate existing files.")

    return results


def main():
    """Main function to execute the sample mapper."""
    parser = argparse.ArgumentParser(
        description="Create GSM to SRR mapping for GSE accession(s)",
        epilog="Examples:\n"
        "  python sample_mapper.py GSE123456\n"
        "  python sample_mapper.py GSE123456 --base-dir /path/to/data\n"
        "  python sample_mapper.py --file gse_list.txt --characteristics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Create mutually exclusive group for GSE input method
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "gse_accession", nargs="?", help="Single GSE accession number (e.g., GSE123456)"
    )
    input_group.add_argument(
        "--file",
        "-f",
        type=str,
        help="Text file containing GSE accessions, one per line",
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="data",
        help="Base directory for data files (default: data). "
        "Mapping files saved to {base-dir}/accessions/, "
        "metadata files saved to {base-dir}/results/{GSE}/",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--characteristics",
        action="store_true",
        help="Also fetch sample characteristics and save to separate JSON files",
    )

    args = parser.parse_args()

    # Determine GSE accessions to process
    if args.gse_accession:
        # Single GSE mode
        gse_accession = args.gse_accession.upper()
        if not gse_accession.startswith("GSE"):
            print(
                f"Error: '{args.gse_accession}' does not appear to be a "
                "valid GSE accession"
            )
            print(
                "GSE accessions should start with 'GSE' followed by numbers "
                "(e.g., GSE123456)"
            )
            return 1
        gse_list = [gse_accession]
        print(f"Processing single GSE: {gse_accession}")
    else:
        # File mode
        gse_list = read_gse_file(args.file)
        if not gse_list:
            print("No valid GSE accessions found. Exiting.")
            return 1

    if args.characteristics:
        print("Sample characteristics will be fetched for all GSE accessions.")

    try:
        if len(gse_list) == 1:
            # Single GSE - use original function for cleaner output
            mapping = create_gsm_to_srr_mapping(
                gse_list[0],
                base_dir=args.base_dir,
                progress=not args.quiet,
                fetch_characteristics=args.characteristics,
            )

            if mapping:
                print("✓ Processing completed successfully!")
                return 0
            else:
                print("✗ No mapping data was generated")
                return 1
        else:
            # Multiple GSE - use batch processing function
            results = process_multiple_gse(
                gse_list,
                base_dir=args.base_dir,
                progress=not args.quiet,
                fetch_characteristics=args.characteristics,
            )

            successful_count = sum(
                1 for r in results.values() if r["status"] in ["success", "skipped"]
            )
            if successful_count > 0:
                print("✓ Batch processing completed!")
                return 0
            else:
                print("✗ No GSE accessions were processed successfully")
                return 1

    except KeyboardInterrupt:
        print("\n✗ Process interrupted by user")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
