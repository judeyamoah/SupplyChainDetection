"""
02b_npm_vulnerable_metadata_enricher.py

Purpose:
Fetch npm registry metadata for known vulnerable npm packages collected from
the OSV bulk export.

Input:
data/malicious/npm_vulnerable_seed.csv

Output:
data/raw/npm_vulnerable_metadata_raw.json

Progress file:
logs/npm_vulnerable_metadata_progress.json

This script helps enrich the minority class for supervised learning.
"""

import csv
import json
import time
from pathlib import Path
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "malicious" / "npm_vulnerable_seed.csv"
OUTPUT_FILE = BASE_DIR / "data" / "raw" / "npm_vulnerable_metadata_raw.json"
PROGRESS_FILE = BASE_DIR / "logs" / "npm_vulnerable_metadata_progress.json"

REGISTRY_BASE_URL = "https://registry.npmjs.org/"

REQUEST_DELAY = 0
MAX_WORKERS = 20

HEADERS = {
    "User-Agent": "MSc-Thesis-Supply-Chain-Research/1.0",
    "Accept": "application/json"
}


# -----------------------------
# Helper Functions
# -----------------------------

def ensure_directories():
    """Create required directories."""
    (BASE_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)


def load_seed_packages():
    """Load known vulnerable package names from CSV."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    packages = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            package_name = row.get("package_name")

            if package_name:
                packages.append({
                    "package_name": package_name,
                    "seed_info": row
                })

    return packages


def load_progress():
    """Load progress file."""
    if PROGRESS_FILE.exists() and PROGRESS_FILE.stat().st_size > 0:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    return {
        "processed": [],
        "failed": []
    }


def save_progress(progress):
    """Save progress atomically."""
    temp_file = PROGRESS_FILE.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(progress, file, indent=4)

    temp_file.replace(PROGRESS_FILE)


def load_existing_results():
    """Load existing metadata if available."""
    if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    return []


def save_results(results):
    """Save metadata atomically."""
    temp_file = OUTPUT_FILE.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    temp_file.replace(OUTPUT_FILE)


def fetch_package_metadata(package_name):
    """
    Fetch full npm metadata for a package.

    Args:
        package_name (str): npm package name.

    Returns:
        dict | None: Metadata JSON.
    """
    encoded_name = quote(package_name, safe="@/")
    url = REGISTRY_BASE_URL + encoded_name

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        if response.status_code == 404:
            return None

        if response.status_code == 429:
            print("Rate limit reached. Sleeping for 10 seconds...")
            time.sleep(10)
            return fetch_package_metadata(package_name)

        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        print(f"Error fetching {package_name}: {error}")
        return None


def attach_seed_info(metadata, seed_info):
    """
    Attach OSV seed information to metadata record.

    This helps preserve data provenance.
    """
    metadata["_seed_label"] = 1
    metadata["_seed_source"] = seed_info.get("source", "OSV_bulk_export")
    metadata["_seed_category"] = seed_info.get("category", "known_vulnerable")
    metadata["_seed_confidence"] = seed_info.get("confidence", "high")
    metadata["_seed_vulnerability_count"] = seed_info.get("vulnerability_count", "")
    metadata["_seed_osv_id"] = seed_info.get("osv_id", "")
    metadata["_seed_aliases"] = seed_info.get("aliases", "")

    return metadata


# -----------------------------
# Main Logic
# -----------------------------

def main():
    ensure_directories()

    seed_packages = load_seed_packages()
    progress = load_progress()
    results = load_existing_results()

    processed = set(progress.get("processed", []))
    failed = set(progress.get("failed", []))

    print("=" * 60)
    print("NPM VULNERABLE METADATA ENRICHMENT STARTED")
    print("=" * 60)
    print(f"Seed vulnerable packages: {len(seed_packages)}")
    print(f"Already processed: {len(processed)}")
    print(f"Previously failed: {len(failed)}")
    print("=" * 60)

   

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_package = {
            executor.submit(
                fetch_package_metadata,
                item["package_name"]
            ): item
            for item in seed_packages
        }

        for future in tqdm(
            as_completed(future_to_package),
            total=len(future_to_package),
            desc="Fetching vulnerable metadata"
        ):
            item = future_to_package[future]
            package_name = item["package_name"]
            seed_info = item["seed_info"]

            try:
                metadata = future.result()

                if metadata is None:
                    failed.add(package_name)
                else:
                    metadata = attach_seed_info(
                        metadata,
                        seed_info
                    )
                    results.append(metadata)

                processed.add(package_name)

            except Exception as error:
                print(
                    f"Error processing {package_name}: "
                    f"{error}"
                )
                failed.add(package_name)

            save_results(results)
            save_progress({
                "processed": sorted(list(processed)),
                "failed": sorted(list(failed))
            })

    print("=" * 60)
    print("NPM VULNERABLE METADATA ENRICHMENT COMPLETE")
    print(f"Metadata records saved: {len(results)}")
    print(f"Failed packages: {len(failed)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
