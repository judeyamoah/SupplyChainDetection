"""
02_npm_metadata_enricher.py

Purpose:
Fetch full npm package metadata for package names collected by 01_npm_collector.py.

Input:
data/raw/npm_packages.json

Output:
data/raw/npm_metadata_raw.json

Progress file:
logs/npm_metadata_progress.json

The script is interruption-safe and resumes from the last processed package.
"""

import json
import time
from pathlib import Path

import requests
from tqdm import tqdm


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "raw" / "npm_packages.json"
OUTPUT_FILE = BASE_DIR / "data" / "raw" / "npm_metadata_raw.json"
PROGRESS_FILE = BASE_DIR / "logs" / "npm_metadata_progress.json"

REGISTRY_BASE_URL = "https://registry.npmjs.org/"

HEADERS = {
    "User-Agent": "MSc-Thesis-Supply-Chain-Research/1.0",
    "Accept": "application/json"
}

REQUEST_DELAY = 0.2


# -----------------------------
# Helper Functions
# -----------------------------

def ensure_directories():
    """Create required directories."""
    (BASE_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)


def load_packages():
    """Load package names collected from npm."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def load_progress():
    """Load processed package list."""
    if PROGRESS_FILE.exists() and PROGRESS_FILE.stat().st_size > 0:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    return {
        "processed": [],
        "failed": []
    }


def save_progress(progress):
    """Save progress file."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as file:
        json.dump(progress, file, indent=4)


def load_existing_results():
    """Load already fetched metadata."""
    if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    return []


def save_results(results):
    """Save metadata results."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)


def fetch_package_metadata(package_name):
    """
    Fetch metadata for a single npm package.

    Args:
        package_name (str): npm package name.

    Returns:
        dict | None: Package metadata.
    """
    url = REGISTRY_BASE_URL + package_name

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        print(f"Error fetching {package_name}: {error}")
        return None


# -----------------------------
# Main Logic
# -----------------------------

def main():
    ensure_directories()

    packages = load_packages()
    progress = load_progress()
    results = load_existing_results()

    processed = set(progress.get("processed", []))
    failed = set(progress.get("failed", []))

    print("=" * 60)
    print("NPM METADATA ENRICHMENT STARTED")
    print("=" * 60)
    print(f"Total package names: {len(packages)}")
    print(f"Already processed: {len(processed)}")
    print(f"Previously failed: {len(failed)}")
    print("=" * 60)

    for package_name in tqdm(packages, desc="Fetching npm metadata"):
        if package_name in processed:
            continue

        metadata = fetch_package_metadata(package_name)

        if metadata is None:
            failed.add(package_name)
        else:
            results.append(metadata)

        processed.add(package_name)

        save_results(results)
        save_progress({
            "processed": sorted(list(processed)),
            "failed": sorted(list(failed))
        })

        time.sleep(REQUEST_DELAY)

    print("=" * 60)
    print("NPM METADATA ENRICHMENT COMPLETE")
    print(f"Metadata records saved: {len(results)}")
    print(f"Failed packages: {len(failed)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()

