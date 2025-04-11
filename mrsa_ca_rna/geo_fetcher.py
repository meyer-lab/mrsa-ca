#!/usr/bin/env python3
"""
GEO Fetcher - A tool for searching and retrieving GEO dataset information

This script provides functionality to:
1. Search for GEO datasets using the NCBI E-utilities API
2. Process lists of GEO accession links
3. Generate sample accession lists for datasets
4. Export dataset information to CSV

Example usage:
  python geo_fetcher.py -s "tuberculosis AND whole blood[Sample Source]"
  python geo_fetcher.py -f geo_links.txt --samples
"""

import argparse
import os
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

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
    """ Make an API request with retry logic for handling rate limits. 
    API key grants us higher rate limits, but let's be curteous.
    Registration of my name and email with NCBI w/Meyer Lab as our org
    can resolve any banned IP issues.

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
    HTTP Response object if successful, else None
        Response object from the requests library
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
    """Performs an esearch for GEO datasets using the NCBI E-utilities API.
    Searches will utilitize the history feature to allow for batch retrieval.
    History allows for large fetches without rate limit and bypasses retmax.

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
    search_url = f"{base_url}esearch.fcgi?db=gds&term={search_term}&retmax={retmax}&usehistory=y&api_key={ncbi_api_key}"

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


def fetch_dataset_details(query_key, web_env, generate_sample_lists=False):
    """Fetches and parses dataset details using the query key and web environment
    obtained from the esearch response. This function retrieves all datasets
    matching the search criteria in a single request, using the history feature
    to bypass the retmax limit.

    Parameters
    ----------
    query_key : str
        Query key from the esearch response
    web_env : str
        web_env from the esearch response
    generate_sample_lists : Bool, optional
        Will create GSEXXXX_sample.txt files containing sample accession numbers
        for each dataset found , by default False. Sample accession lists are used
        by SRA tools for RNA data processing.

    Returns
    -------
    list[dict]
        A list of dictionaries containing details for each dataset.
        See parse_geo_docsum for the keys in each dict
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
                        dataset["samples"] = samples
                        save_sample_accessions_to_file(dataset["accession"], samples)

                results.append(dataset)

        return results

    except ET.ParseError as e:
        print(f"Error parsing XML response: {e}")
        return results


def fetch_dataset_details_individually(id_list, generate_sample_lists=False):
    """If the batch fetch using history fails, this function will fetch each dataset
    individually using its ID. This is a fallback method to ensure we can still retrieve
    dataset details even if the batch fetch fails. However, we must respect the rate
    limits.

    Parameters
    ----------
    id_list : list[str]
        List of GDS IDs to fetch
    generate_sample_lists : Bool, optional
        Whether to print out sample accessions for the datasets, by default False

    Returns
    -------
    list[dict]
        A list of dictionaries containing details for each dataset.
        See parse_geo_docsum for the keys in each dict.
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
                            dataset["samples"] = samples
                            save_sample_accessions_to_file(
                                dataset["accession"], samples
                            )

                    results.append(dataset)

        except ET.ParseError as e:
            print(f"Error parsing XML for dataset {dataset_id}: {e}")

    return results


def fetch_geo_by_accessions(accessions, generate_sample_lists=False):
    """Fetches GEO datasets by their GSE accessions. This process is 2 steps.
    1. Search for the GSE accession to get the UID
    2. Use the UID to fetch complete details
    UIDs are used to retrieve the dataset details, not the GSE accessions.

    Parameters
    ----------
    accessions : list[str]
        List of GEO accessions (e.g., GSE123456)
    generate_sample_lists : Bool, optional
        Whether to print sample accessions to txt file, by default False

    Returns
    -------
    list[dict]
        A list of dictionaries containing details for each dataset.
        See parse_geo_docsum for the keys in each dict.
    """
    results = []
    retry_delay = 1

    for accession in accessions:
        time.sleep(retry_delay)
        print(f"Processing accession: {accession}")

        # Step 1: Search for the GSE accession to get the UID
        search_url = f"{base_url}esearch.fcgi?db=gds&term={accession}[Accession]&api_key={ncbi_api_key}"

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

            # Save example response for debugging
            if not hasattr(fetch_geo_by_accessions, "debug_done"):
                with open("geo_response_example.xml", "wb") as f:
                    f.write(fetch_response.content)
                print("Saved example response to geo_response_example.xml")
                fetch_geo_by_accessions.debug_done = True

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
                            dataset["samples"] = samples
                            save_sample_accessions_to_file(accession, samples)

                    results.append(dataset)
            else:
                print(f"No data found for {accession}")

        except ET.ParseError as e:
            print(f"Error parsing XML for accession {accession}: {e}")

    return results


#############################################
# Data Parsing Functions
#############################################


def parse_geo_docsum(doc):
    """Parses the DocSum XML element to extract dataset metadata.
    The fetch functions use the esummary utility which outputs an XML
    document summary. This function extracts the relevant fields.

    Parameters
    ----------
    doc : Element[str]
        The DocSum XML element containing dataset metadata

    Returns
    -------
    dict
        The parsed dataset metadata as a dictionary
        Keys include 'id', 'accession', 'title', 'summary', 'n_samples',
        'organism', 'type', 'publication_date', 'link', and 'sample_type'
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
    """Extracts the sample accessions from the DocSum XML element.

    Parameters
    ----------
    doc : Element[str]
        The DocSum XML element containing dataset metadata
    series_accession : str
        The GSE accession number for the dataset

    Returns
    -------
    list[dict]
        A list of dictionaries containing sample accessions and titles
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
            sample_info = {
                "accession": accession_item.text,
                "title": title_item.text if title_item is not None else "Unknown",
            }
            samples.append(sample_info)

    return samples


def determine_sample_type(dataset):
    """Fills out the sample type field in the dataset dictionary.
    This function uses the summary and title fields to determine the sample type.

    What's odd is that we can perform an esearch for the sample type as
    [Sample Source] but the esummary does not return this field. We can
    only determine the sample type from the summary and title fields.

    Parameters
    ----------
    dataset : Element[str]
        The dataset dictionary containing metadata

    Returns
    -------
    str
        Sample type as a string.
        Possible values include "Whole Blood", "Peripheral Blood",
        "Blood (unspecified)", "Serum", "Plasma", "Lung/Respiratory",
        "Skin", "Liver", or "Unknown" if not found.
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
    """
    Extract GEO accession number from a URL.
    Example URL: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123456
    """
    match = re.search(r"GSE\d+", url)
    if match:
        return match.group(0)
    return None


def read_geo_links_file(filepath):
    """Reads a text file containing GEO links on each line and extracts the 
    GEO accession numbers. The file can contain comments starting with #.

    Parameters
    ----------
    filepath : str
        Path to the text file containing GEO links
        Each line should contain a GEO link (e.g., GSE123456)

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


def save_sample_accessions_to_file(series_accession, samples):
    """Saves sample accessions to a text file named after the series accession.
    The file will contain the sample accessions and their titles.

    We will later modify this to only include the sample accessions for SRA
    fetching.

    Parameters
    ----------
    series_accession : str
        GSE accession number for the dataset
    samples : list[dict]
        List of dictionaries containing sample accessions and titles
    """
    filename = f"{series_accession}_samples.txt"

    with open(filename, "w") as f:
        f.write(f"# Sample accessions for {series_accession}\n")
        f.write(f"# Total samples: {len(samples)}\n\n")

        for sample in samples:
            f.write(f"{sample['accession']}\t{sample['title']}\n")

    print(f"Saved {len(samples)} sample accessions to {filename}")


def export_datasets_to_csv(datasets, filename="geo_datasets.csv"):
    """Prints out all of the datasets to a CSV file for manual review.
    The CSV will contain the following columns:
    - Accession
    - Number of Samples
    - Sample Type
    - Organism
    - GEO Link
    - Publication Date
    - Title
    - Summary

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
    """
    Main function for running the GEO fetcher.
    Parses command line arguments and executes appropriate workflow.

    The script can be run in two modes:
    1. Search mode: -s <search term>
       - Searches GEO datasets using the specified search term.
       - Generates sample accession lists if --samples is specified.
       - Exports results to a CSV file if --csv is specified.

    2. File mode: -f <file path>
         - Reads GEO links from a file, one per line.
         - Generates sample accession lists if --samples is specified.
         - Exports results to a CSV file if --csv is specified.
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
                query_key, web_env, generate_sample_lists=args.samples
            )

            # If batch fetch returns no results, try individual fetches
            if not datasets:
                print("Batch fetch returned no results. Trying individual fetches...")
                datasets = fetch_dataset_details_individually(
                    id_list, generate_sample_lists=args.samples
                )
        except Exception as e:
            print(f"Error during batch fetch: {e}")
            print("Falling back to individual fetches...")
            datasets = fetch_dataset_details_individually(
                id_list, generate_sample_lists=args.samples
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
            accessions, generate_sample_lists=args.samples
        )

    # Display results
    print("\n--- RESULTS ---")
    print(f"Found {len(datasets)} datasets:")

    for i, dataset in enumerate(datasets, 1):
        print(
            f"\n{i}. {dataset.get('accession', 'Unknown')} - {dataset.get('title', 'No title')}"
        )
        print(f"   Link: {dataset.get('link', 'N/A')}")
        print(f"   Samples: {dataset.get('n_samples', 'Unknown')}")
        print(f"   Sample Type: {dataset.get('sample_type', 'Unknown')}")

    # Export datasets to CSV
    if datasets:
        export_datasets_to_csv(datasets, filename=args.csv)


if __name__ == "__main__":
    main()
