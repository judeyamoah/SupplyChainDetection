"""
01_npm_collector.py

Purpose:
Collect npm package identifiers from the npm CouchDB replication endpoint.

This script retrieves package names from:
https://replicate.npmjs.com/_all_docs

Output:
data/raw/npm_packages.json

Progress file:
logs/npm_progress.json

The script is interruption-safe. If stopped, it resumes from the last
successfully collected package.
"""

import json
import os
import time
from pathlib import Path
from urllib.parse import urlencode

import requests


# -----------------------------
# Configuration
# -----------------------------

BASE_URL = "https://replicate.npmjs.com/_all_docs"
LIMIT = 500
#TARGET_COUNT = 3000
TARGET_COUNT = 1000000

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_FILE = BASE_DIR / "data" / "raw" / "npm_packages.json"
PROGRESS_FILE = BASE_DIR / "logs" / "npm_progress.json"

HEADERS = {
    "User-Agent": "MSc-Thesis-Supply-Chain-Research/1.0",
    "Accept": "application/json"
}


# -----------------------------
# Helper Functions
# -----------------------------

def ensure_directories():
    """Create required project directories if they do not already exist."""
    (BASE_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)


def save_progress(last_key, collected):
    """
    Save current collection progress.

    Args:
        last_key (str): Last package key collected from npm endpoint.
        collected (int): Number of packages collected so far.
    """
    progress = {
        "last_key": last_key,
        "collected": collected
    }

    with open(PROGRESS_FILE, "w", encoding="utf-8") as file:
        json.dump(progress, file, indent=4)


def load_progress():
    """
    Load previous progress if available.

    Returns:
        dict: Progress information containing last_key and collected count.
    """
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    return {
        "last_key": None,
        "collected": 0
    }


def load_existing_packages():
    """
    Load already collected packages if the output file exists.

    Returns:
        list: Existing package names.
    """
    if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    return []


def save_packages(packages):
    """
    Save package list to JSON file.

    Args:
        packages (list): List of npm package names.
    """
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(packages, file, indent=4)


def fetch_batch(startkey=None):
    """
    Fetch a batch of package identifiers from npm replication endpoint.

    Args:
        startkey (str, optional): Last key used for pagination.

    Returns:
        dict | None: JSON response from npm endpoint.
    """
    params = {
        "limit": LIMIT
    }

    if startkey:
        params["startkey"] = json.dumps(startkey)

    url = f"{BASE_URL}?{urlencode(params)}"

    print(f"Requesting: {url}")

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        if response.status_code != 200:
            print(f"HTTP Error {response.status_code}: {response.text[:300]}")
            time.sleep(5)
            return None

        return response.json()

    except requests.RequestException as error:
        print(f"Request failed: {error}")
        time.sleep(5)
        return None


# -----------------------------
# Main Collection Logic
# -----------------------------

def main():
    ensure_directories()

    progress = load_progress()
    last_key = progress.get("last_key")
    collected = progress.get("collected", 0)

    packages = load_existing_packages()

    # Protect against mismatch between progress file and output file
    if len(packages) > collected:
        collected = len(packages)

    print("=" * 60)
    print("NPM PACKAGE COLLECTION STARTED")
    print("=" * 60)
    print(f"Target package count: {TARGET_COUNT}")
    print(f"Already collected: {collected}")
    print(f"Resuming from key: {last_key}")
    print("=" * 60)

    seen_packages = set(packages)

    while collected < TARGET_COUNT:
        data = fetch_batch(last_key)

        if data is None:
            print("No data returned. Retrying...")
            continue

        rows = data.get("rows", [])

        if not rows:
            print("No more rows returned from endpoint.")
            break

        for row in rows:
            package_name = row.get("id")

            if not package_name:
                continue

            # Avoid duplicate caused by CouchDB startkey behaviour
            if package_name in seen_packages:
                continue

            packages.append(package_name)
            seen_packages.add(package_name)
            collected += 1

            if collected >= TARGET_COUNT:
                break

        last_key = rows[-1].get("id")

        save_packages(packages)
        save_progress(last_key, collected)

        print(f"Collected: {collected}/{TARGET_COUNT}")

        # polite delay to avoid hammering the registry
        time.sleep(1)

    print("=" * 60)
    print("NPM PACKAGE COLLECTION COMPLETE")
    print(f"Total packages collected: {len(packages)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
