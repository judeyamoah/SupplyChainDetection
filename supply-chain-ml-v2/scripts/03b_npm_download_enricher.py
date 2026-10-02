"""
03b_npm_download_enricher.py

Purpose:
Fetch npm package download statistics and enrich the dataset with popularity
and exposure indicators.

Input:
data/raw/npm_metadata_raw.json

Output:
data/interim/npm_downloads_enriched.json

Progress file:
logs/npm_downloads_progress.json

API used:
https://api.npmjs.org/downloads/point/last-week/<package>
https://api.npmjs.org/downloads/point/last-month/<package>
"""

import json
import time
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm import tqdm


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# INPUT_FILE = BASE_DIR / "data" / "raw" / "npm_metadata_raw.json"
NORMAL_METADATA_FILE = BASE_DIR / "data" / "raw" / "npm_metadata_raw.json"
VULNERABLE_METADATA_FILE = BASE_DIR / "data" / "raw" / "npm_vulnerable_metadata_raw.json"
OUTPUT_FILE = BASE_DIR / "data" / "interim" / "npm_downloads_enriched.json"
PROGRESS_FILE = BASE_DIR / "logs" / "npm_downloads_progress.json"

DOWNLOADS_BASE_URL = "https://api.npmjs.org/downloads/point"

REQUEST_DELAY = 0.25

HEADERS = {
    "User-Agent": "MSc-Thesis-Supply-Chain-Research/1.0",
    "Accept": "application/json"
}


# -----------------------------
# Helper Functions
# -----------------------------

def ensure_directories():
    """Create required directories."""
    (BASE_DIR / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)


def load_json(path):
    """Load JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    """Save JSON file."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_progress():
    """Load progress."""
    if PROGRESS_FILE.exists() and PROGRESS_FILE.stat().st_size > 0:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    return {
        "processed": [],
        "failed": []
    }


def extract_package_names(metadata_records):
    """Extract package names from npm metadata."""
    package_names = []

    for record in metadata_records:
        name = record.get("name")
        if name:
            package_names.append(name)

    return package_names


def fetch_download_count(package_name, period):
    """
    Fetch npm download count for a package over a specified period.

    Args:
        package_name (str): npm package name.
        period (str): last-week or last-month.

    Returns:
        int: Download count.
    """
    encoded_name = quote(package_name, safe="@/")
    url = f"{DOWNLOADS_BASE_URL}/{period}/{encoded_name}"

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        if response.status_code == 404:
            return 0

        if response.status_code == 429:
            print("Rate limit reached. Sleeping for 10 seconds...")
            time.sleep(10)
            return fetch_download_count(package_name, period)

        response.raise_for_status()

        data = response.json()
        return int(data.get("downloads", 0))

    except Exception as error:
        print(f"Download fetch failed for {package_name} ({period}): {error}")
        return None


# -----------------------------
# Main Logic
# -----------------------------

def main():
    ensure_directories()

    # metadata_records = load_json(INPUT_FILE)
    # package_names = extract_package_names(metadata_records)
    normal_metadata = load_json(NORMAL_METADATA_FILE)

    if VULNERABLE_METADATA_FILE.exists() and VULNERABLE_METADATA_FILE.stat().st_size > 0:
        vulnerable_metadata = load_json(VULNERABLE_METADATA_FILE)
    else:
        vulnerable_metadata = []

    metadata_records = normal_metadata + vulnerable_metadata
    package_names = list(set(extract_package_names(metadata_records)))

    progress = load_progress()
    processed = set(progress.get("processed", []))
    failed = set(progress.get("failed", []))

    if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        results = load_json(OUTPUT_FILE)
    else:
        results = []

    print("=" * 60)
    print("NPM DOWNLOAD ENRICHMENT STARTED")
    print("=" * 60)
    print(f"Total packages: {len(package_names)}")
    print(f"Already processed: {len(processed)}")
    print("=" * 60)

    for package_name in tqdm(package_names, desc="Fetching npm downloads"):
        if package_name in processed:
            continue

        weekly_downloads = fetch_download_count(package_name, "last-week")
        time.sleep(REQUEST_DELAY)

        monthly_downloads = fetch_download_count(package_name, "last-month")
        time.sleep(REQUEST_DELAY)

        if weekly_downloads is None or monthly_downloads is None:
            failed.add(package_name)
            weekly_downloads = 0 if weekly_downloads is None else weekly_downloads
            monthly_downloads = 0 if monthly_downloads is None else monthly_downloads

        record = {
            "package_name": package_name,
            "weekly_downloads": weekly_downloads,
            "monthly_downloads": monthly_downloads
        }

        results.append(record)
        processed.add(package_name)

        save_json(OUTPUT_FILE, results)
        save_json(PROGRESS_FILE, {
            "processed": sorted(list(processed)),
            "failed": sorted(list(failed))
        })

    print("=" * 60)
    print("NPM DOWNLOAD ENRICHMENT COMPLETE")
    print(f"Records saved: {len(results)}")
    print(f"Failed packages: {len(failed)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
