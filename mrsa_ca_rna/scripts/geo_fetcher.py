#!/usr/bin/env python3
"""
GEO Fetcher - A tool for searching and retrieving GEO dataset information

This script provides functionality to:
1. Search for GEO datasets using the NCBI E-utilities API
2. Process lists of GEO accession links
3. Generate sample accession lists for datasets
4. Export dataset information to CSV with enhanced metadata

Example usage:
  python geo_fetcher.py -s "tuberculosis AND whole blood[Sample Source]"
  python geo_fetcher.py -f geo_links.txt --samples --output-dir data
"""

import argparse
import os
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import GEOparse
import pandas as pd
import requests

#############################################
# Configuration
#############################################

# NCBI E-utilities base URL
base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# NCBI API key for E-utilities
ncbi_api_key = "7f7ac246396cc1bd1f47c090de5ddebeb709"

#############################################
# E-utilities API Functions
#############################################


def make_api_request(url, retry_delay=1, max_retries=3):
    """Make an API request with retry logic for handling rate limits.
    
    Parameters
    ----------
    url : str
        API url for the request
    retry_delay : float, optional
        Seconds to increase the delay if failure, by default 1
    max_retries : int, optional
        Maximum number of retry attempts to fetch, by default 3

    Returns
    -------
    requests.Response or None
        Response object from the requests library if successful, else None
    """
    for retry in range(max_retries):
        response = requests.get(url)

        if response.status_code == 200:
            return response
        elif response.status_code == 429:
            print(f"Rate limit hit, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay * (retry + 1))
            retry_delay = min(retry_delay * 2, 10)  # Exponential backoff
        else:
            print(f"Error: HTTP {response.status_code}")
            if response.content:
                print(f"Error details: {response.content[:200]}...")
            break

    return None


def search_geo_datasets(query, retmax=20):
    """Perform an esearch for GEO datasets using the NCBI E-utilities API.
    
    Parameters
    ----------
    query : str
        GDS search term. Visit https://www.ncbi.nlm.nih.gov/geo/info/qqtutorial.html
        for more information on search terms.
    retmax : int, optional
        Maximum dataset IDs to return, by default 20

    Returns
    -------
    tuple[list[str], str, str]
        A tuple containing:
        - A list of GEO dataset IDs (GDS numbers)
        - The query key for the search
        - The web environment for the search
    """
    # URL Encode search term as required by E-utilities
    search_term = quote_plus(query)

    # Prepare search URL
    search_url = (
        f"{base_url}esearch.fcgi?db=gds&term={search_term}&retmax={retmax}"
        f"&usehistory=y&api_key={ncbi_api_key}"
    )

    print(f"Executing search: {query}")
    response = make_api_request(search_url)

    if not response:
        return [], None, None

    # Parse the XML response
    try:
        root = ET.fromstring(response.content)

        # Get the query key and web environment
        query_key = (
            root.find("QueryKey").text if root.find("QueryKey") is not None else None
        )
        web_env = root.find("WebEnv").text if root.find("WebEnv") is not None else None

        # Extract the GEO IDs
        id_list = [id_elem.text for id_elem in root.findall(".//Id")]

        return id_list, query_key, web_env

    except ET.ParseError as e:
        print(f"Error parsing XML response: {e}")
        return [], None, None


def fetch_dataset_details(query_key, web_env, generate_sample_lists=False, output_dir="data"):
    """Fetch dataset details using the history feature.
    
    Parameters
    ----------
    query_key : str
        Query key from the search
    web_env : str
        Web environment from the search
    generate_sample_lists : bool, optional
        Whether to generate sample lists, by default False
    output_dir : str, optional
        Directory for output files, by default "data"
        
    Returns
    -------
    list[dict]
        List of dictionaries containing dataset details
    """
    results = []

    # Prepare fetch URL
    fetch_url = f"{base_url}esummary.fcgi?db=gds&query_key={query_key}&WebEnv={web_env}&api_key={ncbi_api_key}"
    
    response = make_api_request(fetch_url)
    if not response:
        return results

    # Parse the XML response
    try:
        root = ET.fromstring(response.content)

        # Process each dataset
        for doc in root.findall(".//DocSum"):
            dataset = parse_geo_docsum(doc)
            if dataset:
                # Extract sample accessions if requested
                if generate_sample_lists:
                    samples = extract_sample_accessions(doc, dataset["accession"])
                    if samples:
                        # Get detailed metadata and SRA accessions
                        enhanced_samples = process_samples(dataset["accession"], samples)
                        dataset["samples"] = enhanced_samples
                        save_sample_metadata_to_csv(dataset["accession"], enhanced_samples, output_dir)

                results.append(dataset)

        return results

    except ET.ParseError as e:
        print(f"Error parsing XML response: {e}")
        return results


def fetch_dataset_details_individually(id_list, generate_sample_lists=False, output_dir="data"):
    """Fetch each dataset individually if batch history method fails.
    
    Parameters
    ----------
    id_list : list[str]
        List of GDS IDs to fetch
    generate_sample_lists : bool, optional
        Whether to generate enhanced sample metadata, by default False
    output_dir : str, optional
        Directory for output files, by default "data"

    Returns
    -------
    list[dict]
        A list of dictionaries containing details for each dataset.
    """
    results = []
    retry_delay = 1

    for dataset_id in id_list:
        # Add a small delay between requests
        time.sleep(retry_delay)

        # Prepare fetch URL for this specific dataset
        fetch_url = (
            f"{base_url}esummary.fcgi?db=gds&id={dataset_id}&api_key={ncbi_api_key}"
        )

        response = make_api_request(fetch_url, retry_delay)
        if not response:
            continue

        # Parse the XML response
        try:
            root = ET.fromstring(response.content)

            # Process the dataset
            doc = root.find(".//DocSum")
            if doc is not None:
                dataset = parse_geo_docsum(doc)
                if dataset:
                    # Extract sample accessions if requested
                    if generate_sample_lists:
                        samples = extract_sample_accessions(doc, dataset["accession"])
                        if samples:
                            # Use the enhanced processing like other functions
                            enhanced_samples = process_samples(dataset["accession"], samples)
                            dataset["samples"] = enhanced_samples
                            save_sample_metadata_to_csv(dataset["accession"], enhanced_samples, output_dir)

                    results.append(dataset)

        except ET.ParseError as e:
            print(f"Error parsing XML for dataset {dataset_id}: {e}")

    return results


def fetch_geo_by_accessions(accessions, generate_sample_lists=False, output_dir="data"):
    """Fetch GEO datasets by their GSE accessions using a two-step process.
    
    Parameters
    ----------
    accessions : list[str]
        List of GEO accessions (e.g., GSE123456)
    generate_sample_lists : bool, optional
        Whether to generate enhanced sample metadata, by default False
    output_dir : str, optional
        Directory for output files, by default "data"

    Returns
    -------
    list[dict]
        A list of dictionaries containing details for each dataset.
    """
    results = []
    retry_delay = 0.5
    for accession in accessions:
        time.sleep(retry_delay)
        print(f"Processing accession: {accession}")

        # Step 1: Search for the GSE accession to get the UID
        search_url = (
            f"{base_url}esearch.fcgi?db=gds&term={accession}[Accession]"
            f"&api_key={ncbi_api_key}"
        )

        search_response = make_api_request(search_url, retry_delay)
        if not search_response:
            print(f"Failed to find UID for accession {accession}")
            continue

        # Parse the search response to get the UID
        try:
            search_root = ET.fromstring(search_response.content)
            id_elems = search_root.findall(".//Id")

            if not id_elems:
                print(f"No UID found for {accession}")
                continue

            uid = id_elems[0].text
            print(f"Found UID {uid} for accession {accession}")

            # Step 2: Use the UID to fetch complete details
            fetch_url = (
                f"{base_url}esummary.fcgi?db=gds&id={uid}&api_key={ncbi_api_key}"
            )

            fetch_response = make_api_request(fetch_url, retry_delay)
            if not fetch_response:
                continue

            # Parse the XML response
            root = ET.fromstring(fetch_response.content)

            # Process the dataset
            doc = root.find(".//DocSum")
            if doc is not None:
                dataset = parse_geo_docsum(doc)
                if dataset:
                    # Extract sample accessions if requested
                    if generate_sample_lists:
                        samples = extract_sample_accessions(doc, accession)
                        if samples:
                            # Use the same enhanced processing as the search mode
                            enhanced_samples = process_samples(accession, samples)
                            dataset["samples"] = enhanced_samples
                            save_sample_metadata_to_csv(accession, enhanced_samples, output_dir)

                    results.append(dataset)
            else:
                print(f"No data found for {accession}")

        except ET.ParseError as e:
            print(f"Error parsing XML for accession {accession}: {e}")

    return results


def gsm_to_sra(gsm_accession, retry_delay=1, get_experiment_ids=True):
    """Convert a GSM accession to SRA accessions for use with sra-tools.
    
    Parameters
    ----------
    gsm_accession : str
        The GSM accession number
    retry_delay : float, optional
        Delay between API requests in seconds, by default 1
    get_experiment_ids : bool, optional
        Whether to return SRX (experiment) IDs rather than SRR (run) IDs, by default True
        
    Returns
    -------
    dict
        Dictionary with keys 'srx_accessions' and 'srr_accessions'
    """
    time.sleep(retry_delay)  # Respect rate limits
    
    # Search for the SRA entry linked to this GSM
    search_url = f"{base_url}esearch.fcgi?db=sra&term={gsm_accession}&api_key={ncbi_api_key}"
    
    response = make_api_request(search_url, retry_delay)
    if not response:
        return {"srx_accessions": [], "srr_accessions": []}
    
    try:
        root = ET.fromstring(response.content)
        count = int(root.find("Count").text) if root.find("Count") is not None else 0
        
        if count == 0:
            print(f"No SRA entries found for {gsm_accession}")
            return {"srx_accessions": [], "srr_accessions": []}
        
        # Get the SRA UIDs
        sra_uids = [id_elem.text for id_elem in root.findall(".//Id")]
        
        if not sra_uids:
            return {"srx_accessions": [], "srr_accessions": []}
        
        # Use efetch with runinfo format
        time.sleep(retry_delay)
        fetch_url = f"{base_url}efetch.fcgi?db=sra&id={','.join(sra_uids)}&rettype=runinfo&api_key={ncbi_api_key}"
        
        fetch_response = make_api_request(fetch_url, retry_delay)
        if not fetch_response:
            return {"srx_accessions": [], "srr_accessions": []}
        
        # Parse the XML response
        try:
            runinfo_root = ET.fromstring(fetch_response.text)
            
            # Extract SRR and SRX accessions
            srr_accessions = []
            srx_accessions = set()  # Use a set to avoid duplicates
            
            for row in runinfo_root.findall(".//Row"):
                run_elem = row.find("Run")
                exp_elem = row.find("Experiment")
                
                if run_elem is not None and run_elem.text:
                    srr_accessions.append(run_elem.text)
                    
                if exp_elem is not None and exp_elem.text:
                    srx_accessions.add(exp_elem.text)
            
            srx_list = list(srx_accessions)
            
            print(f"Found {len(srx_list)} SRX accessions for {gsm_accession}: {srx_list}")
            print(f"Found {len(srr_accessions)} SRR accessions for {gsm_accession}")
            
            return {
                "srx_accessions": srx_list,
                "srr_accessions": srr_accessions
            }
            
        except ET.ParseError as e:
            # Fall back to CSV parsing
            print(f"XML parsing failed, trying CSV format: {e}")
            lines = fetch_response.text.strip().split("\n")
            if len(lines) < 2:  # Need at least header + 1 data row
                return {"srx_accessions": [], "srr_accessions": []}
            
            # Parse CSV for both Run and Experiment columns
            header = lines[0].split(",")
            try:
                results = {"srx_accessions": set(), "srr_accessions": []}
                
                if "Run" in header:
                    run_index = header.index("Run")
                    results["srr_accessions"] = [
                        line.split(",")[run_index] 
                        for line in lines[1:] 
                        if len(line.split(",")) > run_index
                    ]
                
                if "Experiment" in header:
                    exp_index = header.index("Experiment")
                    for line in lines[1:]:
                        parts = line.split(",")
                        if len(parts) > exp_index:
                            results["srx_accessions"].add(parts[exp_index])
                
                # Convert set to list for srx_accessions
                results["srx_accessions"] = list(results["srx_accessions"])
                
                print(f"Found {len(results['srx_accessions'])} SRX accessions for {gsm_accession}")
                print(f"Found {len(results['srr_accessions'])} SRR accessions for {gsm_accession}")
                
                return results
                
            except (ValueError, IndexError):
                print(f"Could not parse runinfo format for {gsm_accession}")
                return {"srx_accessions": [], "srr_accessions": []}
                
    except Exception as e:
        print(f"Error processing SRA data for {gsm_accession}: {str(e)}")
        return {"srx_accessions": [], "srr_accessions": []}


#############################################
# Data Parsing Functions
#############################################


def parse_geo_docsum(doc):
    """Parse the DocSum XML element to extract dataset metadata.
    
    Parameters
    ----------
    doc : Element
        The DocSum XML element containing dataset metadata

    Returns
    -------
    dict or None
        The parsed dataset metadata as a dictionary, or None if no accession found
    """
    dataset = {}

    # Extract the UID (ID)
    id_elem = doc.find("Id")
    if id_elem is not None:
        dataset["id"] = id_elem.text

    # Process each Item element
    for item in doc.findall("./Item"):
        name = item.get("Name")

        # Extract common fields
        if name == "Accession":
            dataset["accession"] = item.text
        elif name == "title":
            dataset["title"] = item.text
        elif name == "summary":
            dataset["summary"] = item.text
        elif name == "n_samples":
            dataset["n_samples"] = item.text
        elif name == "taxon":
            dataset["organism"] = item.text
        elif name == "gdsType":
            dataset["type"] = item.text
        elif name == "PDAT":
            dataset["publication_date"] = item.text

    # Only process if we found an accession
    if "accession" in dataset:
        # Construct GEO link
        dataset["link"] = (
            f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={dataset['accession']}"
        )

        # Extract sample type
        dataset["sample_type"] = determine_sample_type(dataset)

        return dataset

    return None


def extract_sample_accessions(doc, series_accession):
    """Extract basic GSM sample accessions from a GSE series document.
    
    Parameters
    ----------
    doc : Element
        DocSum XML element for the GSE series
    series_accession : str
        The GSE accession number
        
    Returns
    -------
    list[dict]
        List of dictionaries with basic sample information
    """
    samples = []
    
    # Find the Samples list item
    samples_item = doc.find("./Item[@Name='Samples']")
    if samples_item is None:
        print(f"No samples found for {series_accession}")
        return samples
        
    # Extract each sample accession
    for sample in samples_item.findall("./Item[@Name='Sample']"):
        accession_item = sample.find("./Item[@Name='Accession']")
        title_item = sample.find("./Item[@Name='Title']")
        
        if accession_item is not None:
            gsm_accession = accession_item.text
            sample_info = {
                "gsm_accession": gsm_accession,
                "title": title_item.text if title_item is not None else "Unknown"
            }
            samples.append(sample_info)
            
    print(f"Found {len(samples)} GSM accessions for {series_accession}")
    return samples


def fetch_gsm_metadata(gsm_accession, retry_delay=1):
    """Fetch detailed metadata for a GSM sample using GEOparse library.
    
    Parameters
    ----------
    gsm_accession : str
        The GSM accession number
    retry_delay : float, optional
        Delay between API requests, by default 1
        
    Returns
    -------
    dict
        Dictionary containing sample metadata including characteristics
    """
    print(f"Fetching metadata for {gsm_accession} with GEOparse...")
    
    try:
        # Create a temporary directory for GEOparse files
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # GEOparse will download the GSM data to the temp directory
            gsm = GEOparse.get_GEO(geo=gsm_accession, silent=True, destdir=tmp_dir)
            
            metadata = {
                "characteristics": {},
                "platform": gsm.metadata.get("platform_id", [""])[0],
                "instrument": gsm.metadata.get("instrument_model", [""])[0],
                "library_strategy": gsm.metadata.get("library_strategy", [""])[0],
                "submission_date": gsm.metadata.get("submission_date", [""])[0],
                "channel_count": gsm.metadata.get("channel_count", [""])[0]
            }
            
            # Extract characteristics into a flat dictionary
            if "characteristics_ch1" in gsm.metadata:
                for char_entry in gsm.metadata["characteristics_ch1"]:
                    # Format is typically "key: value"
                    if ":" in char_entry:
                        key, value = char_entry.split(":", 1)
                        safe_key = key.strip().replace(" ", "_").replace("-", "_").lower()
                        metadata["characteristics"][safe_key] = value.strip()
            
            # DEBUG CODE - Uncomment for troubleshooting
            # if not hasattr(fetch_gsm_metadata, "debug_done"):
            #     import json
            #     with open("gsm_metadata_example.json", "w") as f:
            #         # Convert metadata keys to strings since some might be complex objects
            #         simple_metadata = {k: str(v) for k, v in gsm.metadata.items()}
            #         json.dump(simple_metadata, f, indent=2)
            #     print("Saved example GSM metadata to gsm_metadata_example.json")
            #     fetch_gsm_metadata.debug_done = True
                
            # When this block ends, the temporary directory and its contents are deleted
            
        return metadata
        
    except Exception as e:
        print(f"Error fetching metadata for {gsm_accession} with GEOparse: {str(e)}")
        # Fall back to empty characteristics if GEOparse fails
        return {"characteristics": {}}


def process_samples(gse_accession, samples):
    """Process a list of samples to get full metadata and SRA accessions.
    
    Parameters
    ----------
    gse_accession : str
        The GSE accession number
    samples : list[dict]
        List of dictionaries with basic sample information
        
    Returns
    -------
    list[dict]
        Enhanced sample list with metadata and SRA accessions
    """
    print(f"Processing {len(samples)} samples for {gse_accession}")
    enhanced_samples = []
    
    for i, sample in enumerate(samples, 1):
        gsm_accession = sample["gsm_accession"]
        print(f"Processing sample {i}/{len(samples)}: {gsm_accession}")
        
        # Get detailed metadata using GEOparse
        metadata = fetch_gsm_metadata(gsm_accession)
        
        # Get SRA accessions
        sra_data = gsm_to_sra(gsm_accession)
        
        # Combine all information
        enhanced_sample = {
            "gsm_accession": gsm_accession,
            "title": sample["title"],
            "srx_accessions": ",".join(sra_data.get("srx_accessions", [])),
            "srr_accessions": ",".join(sra_data.get("srr_accessions", [])),
            "platform": metadata.get("platform", ""),
            "instrument": metadata.get("instrument", ""),
            "library_strategy": metadata.get("library_strategy", ""),
            "submission_date": metadata.get("submission_date", "")
        }
        
        # Add characteristics
        for key, value in metadata.get("characteristics", {}).items():
            # Create safe column name
            safe_key = key.replace(" ", "_").replace("-", "_").lower()
            enhanced_sample[f"char_{safe_key}"] = value
        
        enhanced_samples.append(enhanced_sample)
        
    return enhanced_samples


def determine_sample_type(dataset):
    """Determine sample type from dataset metadata by looking for key terms.
    
    Parameters
    ----------
    dataset : dict
        The dataset dictionary containing metadata

    Returns
    -------
    str
        Sample type as a string
    """
    sample_type = "Unknown"

    if "summary" in dataset:
        summary_text = dataset.get("summary", "").lower()
        title_text = dataset.get("title", "").lower()

        # Dictionary of sample types and their identifying terms
        sample_types = {
            "Whole Blood": ["whole blood", "blood whole"],
            "Peripheral Blood": ["peripheral blood", "pbmc"],
            "Blood (unspecified)": ["blood"],
            "Serum": ["serum"],
            "Plasma": ["plasma"],
            "Lung/Respiratory": ["lung", "respiratory"],
            "Skin": ["skin"],
            "Liver": ["liver"],
        }

        # Check each sample type
        for type_name, terms in sample_types.items():
            if any(term in summary_text or term in title_text for term in terms):
                sample_type = type_name
                break

    return sample_type


#############################################
# Utility Functions
#############################################


def extract_geo_accession(url):
    """Extract GEO accession number from a URL.
    
    Parameters
    ----------
    url : str
        URL potentially containing a GEO accession
        
    Returns
    -------
    str or None
        Extracted GEO accession number or None if not found
    """
    match = re.search(r"GSE\d+", url)
    if match:
        return match.group(0)
    return None


def read_geo_links_file(filepath):
    """Read a text file containing GEO links and extract accession numbers.
    
    Parameters
    ----------
    filepath : str
        Path to the text file containing GEO links

    Returns
    -------
    list[str]
        A list of GEO accession numbers extracted from the file
    """
    accessions = []

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return accessions

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):  # Skip empty lines and comments
                accession = extract_geo_accession(line)
                if accession:
                    accessions.append(accession)
                else:
                    print(f"Warning: Could not extract GEO accession from {line}")

    return accessions


def save_sample_metadata_to_csv(gse_accession, samples, output_dir="data"):
    """Save sample metadata to a CSV file in the specified output directory.
    
    Parameters
    ----------
    gse_accession : str
        The GSE accession number
    samples : list[dict]
        List of dictionaries with sample information
    output_dir : str, optional
        Base directory for outputs, by default "data"
    """
    if not samples:
        print(f"No samples to save for {gse_accession}")
        return
    
    # Create output directories if they don't exist
    metadata_dir = os.path.join(output_dir, "metadata")
    accessions_dir = os.path.join(output_dir, "accessions")
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(accessions_dir, exist_ok=True)
    
    # Path for CSV file
    csv_path = os.path.join(metadata_dir, f"{gse_accession}_samples.csv")
    
    # Convert to DataFrame
    df = pd.DataFrame(samples)
    
    # Reorder columns to put key fields first
    cols = df.columns.tolist()
    key_cols = ["gsm_accession", "title", "srx_accessions", "srr_accessions"]
    reordered_cols = [col for col in key_cols if col in cols] + [col for col in cols if col not in key_cols]
    df = df[reordered_cols]
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(samples)} samples with metadata to {csv_path}")
    
    # Create SRR accessions file with GSM associations
    srr_path = os.path.join(accessions_dir, f"{gse_accession}_srr_accessions.txt")
    with open(srr_path, "w") as f:
        for sample in samples:
            gsm_accession = sample.get("gsm_accession", "")
            srr_list = sample.get("srr_accessions", "").split(",")
            srr_accessions = [srr.strip() for srr in srr_list if srr.strip()]
            
            if srr_accessions:  # Only write if there are SRR accessions
                f.write(f"{gsm_accession} | {' '.join(srr_accessions)}\n")
    
    print(f"Saved GSM-SRR mappings to {srr_path}")


def export_datasets_to_csv(datasets, filename="geo_datasets.csv"):
    """Export datasets to a CSV file for manual review.
    
    Parameters
    ----------
    datasets : list[dict]
        List of dictionaries containing dataset metadata
    filename : str, optional
        File path to export CSV, by default "geo_datasets.csv"
    """
    if not datasets:
        print("No datasets to export.")
        return

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(datasets)

    # Select and rename columns for the CSV
    columns_to_include = {
        "accession": "Accession",
        "n_samples": "Number of Samples",
        "sample_type": "Sample Type",
        "organism": "Organism",
        "link": "GEO Link",
        "publication_date": "Publication Date",
        "title": "Title",
        "summary": "Summary",
    }

    # Keep only the columns that exist in the DataFrame
    available_columns = {k: v for k, v in columns_to_include.items() if k in df.columns}

    # Select and rename columns
    if available_columns:
        df = df[list(available_columns.keys())]
        df = df.rename(columns=available_columns)

    # Order the rows by largest number of samples
    if "Number of Samples" in df.columns:
        # Convert to numeric first (in case it's stored as string)
        df["Number of Samples"] = pd.to_numeric(
            df["Number of Samples"], errors="coerce"
        )
        df = df.sort_values(by="Number of Samples", ascending=False)

    # Export to CSV
    df.to_csv(filename, index=False)
    print(f"Exported {len(datasets)} datasets to {filename}")


#############################################
# Main Program
#############################################


def main():
    """Main function to execute the GEO fetcher workflow.
    
    Parses command line arguments and performs the appropriate workflow based on
    whether the user specified a search query or a file with GEO links.
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Fetch GEO dataset information")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--search", help="Search query for GEO datasets")
    group.add_argument("-f", "--file", help="File containing GEO links, one per line")
    parser.add_argument(
        "-m",
        "--max",
        type=int,
        default=20,
        help="Maximum number of results to return (for search only)",
    )
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Generate sample accession lists for each dataset",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="geo_datasets.csv",
        help="Output CSV filename (default: geo_datasets.csv)",
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="mrsa_ca_rna/data",
        help="Directory for output files (default: data)"
    )
    args = parser.parse_args()

    datasets = []

    # Process based on input mode
    if args.search:
        # Search mode
        search_query = args.search
        print(f"Searching GEO for: {search_query}")

        # Search for datasets
        id_list, query_key, web_env = search_geo_datasets(search_query, args.max)

        if not id_list:
            print("No datasets found matching the criteria.")
            return

        print(f"Found {len(id_list)} potential datasets.")

        # Fetch dataset details
        try:
            datasets = fetch_dataset_details(
                query_key, web_env, generate_sample_lists=args.samples, output_dir=args.output_dir
            )

            # If batch fetch returns no results, try individual fetches
            if not datasets:
                print("Batch fetch returned no results. Trying individual fetches...")
                datasets = fetch_dataset_details_individually(
                    id_list, generate_sample_lists=args.samples, output_dir=args.output_dir
                )
        except Exception as e:
            print(f"Error during batch fetch: {e}")
            print("Falling back to individual fetches...")
            datasets = fetch_dataset_details_individually(
                id_list, generate_sample_lists=args.samples, output_dir=args.output_dir
            )

    elif args.file:
        # File mode
        filepath = args.file
        print(f"Processing GEO links from file: {filepath}")

        # Read accessions from file
        accessions = read_geo_links_file(filepath)

        if not accessions:
            print("No valid GEO accessions found in the file.")
            return

        print(f"Found {len(accessions)} GEO accessions.")

        # Fetch dataset details by accessions
        datasets = fetch_geo_by_accessions(
            accessions, generate_sample_lists=args.samples, output_dir=args.output_dir
        )

    # Display results
    print("\n--- RESULTS ---")
    print(f"Found {len(datasets)} datasets:")

    for i, dataset in enumerate(datasets, 1):
        print(
            f"\n{i}. {dataset.get('accession', 'Unknown')} - "
            f"{dataset.get('title', 'No title')}"
        )
        print(f"   Link: {dataset.get('link', 'N/A')}")
        print(f"   Samples: {dataset.get('n_samples', 'Unknown')}")
        print(f"   Sample Type: {dataset.get('sample_type', 'Unknown')}")

    # Export datasets to CSV
    if datasets:
        export_datasets_to_csv(datasets, filename=args.csv)


if __name__ == "__main__":
    main()
