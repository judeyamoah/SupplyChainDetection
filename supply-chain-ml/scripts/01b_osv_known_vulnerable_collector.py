"""
01b_osv_known_vulnerable_collector.py

Purpose:
Collect known vulnerable npm package names from the OSV bulk vulnerability
database and create a vulnerable seed dataset for class balancing.

Main output:
data/malicious/npm_vulnerable_seed.csv
data/malicious/npm_vulnerable_seed.json

This script helps address class imbalance by adding verified vulnerable npm
packages to the training dataset.

Data source:
OSV bulk data export for npm:
https://storage.googleapis.com/osv-vulnerabilities/npm/all.zip
"""

import csv
import io
import json
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

OSV_NPM_ZIP_URL = "https://storage.googleapis.com/osv-vulnerabilities/npm/all.zip"

OUTPUT_CSV = BASE_DIR / "data" / "malicious" / "npm_vulnerable_seed.csv"
OUTPUT_JSON = BASE_DIR / "data" / "malicious" / "npm_vulnerable_seed.json"
RAW_ZIP_FILE = BASE_DIR / "data" / "raw" / "osv_npm_all.zip"

MAX_VULNERABLE_PACKAGES = 0 

HEADERS = {
    "User-Agent": "MSc-Thesis-Supply-Chain-Research/1.0",
    "Accept": "application/zip,application/json,*/*"
}


# -----------------------------
# Helper Functions
# -----------------------------

def ensure_directories():
    """Create required directories."""
    (BASE_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data" / "malicious").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)


def download_osv_zip():
    """
    Download the OSV npm vulnerability bulk dataset.

    Returns:
        bytes: ZIP file content.
    """
    print("Downloading OSV npm vulnerability dataset...")
    print(OSV_NPM_ZIP_URL)

    response = requests.get(
        OSV_NPM_ZIP_URL,
        headers=HEADERS,
        timeout=120
    )

    response.raise_for_status()

    RAW_ZIP_FILE.write_bytes(response.content)

    print(f"OSV ZIP saved to: {RAW_ZIP_FILE}")
    print(f"Downloaded size: {len(response.content) / (1024 * 1024):.2f} MB")

    return response.content


def load_osv_zip():
    """
    Load OSV ZIP from disk if it already exists, otherwise download it.

    Returns:
        bytes: ZIP file content.
    """
    if RAW_ZIP_FILE.exists() and RAW_ZIP_FILE.stat().st_size > 0:
        print(f"Using existing OSV ZIP file: {RAW_ZIP_FILE}")
        return RAW_ZIP_FILE.read_bytes()

    return download_osv_zip()


def extract_package_records_from_vulnerability(vuln_record):
    """
    Extract npm package records from a single OSV vulnerability record.

    Args:
        vuln_record (dict): OSV vulnerability JSON object.

    Returns:
        list[dict]: Extracted package records.
    """
    records = []

    osv_id = vuln_record.get("id", "")
    aliases = vuln_record.get("aliases", []) or []
    published = vuln_record.get("published", "")
    modified = vuln_record.get("modified", "")

    affected_items = vuln_record.get("affected", []) or []

    for affected in affected_items:
        package = affected.get("package", {}) or {}

        ecosystem = package.get("ecosystem", "")
        package_name = package.get("name", "")

        if ecosystem.lower() != "npm":
            continue

        if not package_name:
            continue

        severity_items = vuln_record.get("severity", []) or []
        severity_scores = []

        for severity in severity_items:
            severity_type = severity.get("type", "")
            score = severity.get("score", "")

            if severity_type or score:
                severity_scores.append(f"{severity_type}:{score}")

        has_cve = any(str(alias).startswith("CVE-") for alias in aliases)
        has_ghsa = any(str(alias).startswith("GHSA-") for alias in aliases)

        records.append({
            "package_name": package_name,
            "osv_id": osv_id,
            "aliases": ";".join(aliases),
            "has_cve_alias": 1 if has_cve else 0,
            "has_ghsa_alias": 1 if has_ghsa else 0,
            "published": published,
            "modified": modified,
            "severity": ";".join(severity_scores),
            "source": "OSV_bulk_export",
            "category": "known_vulnerable",
            "confidence": "high",
            "label": 1
        })

    return records


def parse_osv_zip(zip_bytes):
    """
    Parse OSV ZIP and extract vulnerable npm packages.

    Args:
        zip_bytes (bytes): ZIP file content.

    Returns:
        list[dict]: Deduplicated vulnerable package records.
    """
    extracted_records = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_file:
        json_files = [
            name for name in zip_file.namelist()
            if name.endswith(".json")
        ]

        print(f"OSV vulnerability JSON files found: {len(json_files)}")

        for file_name in tqdm(json_files, desc="Parsing OSV vulnerabilities"):
            try:
                with zip_file.open(file_name) as file:
                    vuln_record = json.loads(file.read().decode("utf-8"))

                package_records = extract_package_records_from_vulnerability(
                    vuln_record
                )

                extracted_records.extend(package_records)

            except Exception as error:
                print(f"Error parsing {file_name}: {error}")
                continue

    return deduplicate_by_package(extracted_records)


def deduplicate_by_package(records):
    """
    Deduplicate records by package name.

    If one package appears in multiple vulnerabilities, the first occurrence
    is retained and vulnerability_count is incremented.
    """
    lookup = {}

    for record in records:
        package_name = record["package_name"]

        if package_name not in lookup:
            record["vulnerability_count"] = 1
            lookup[package_name] = record
        else:
            existing = lookup[package_name]
            existing["vulnerability_count"] += 1

            if record.get("has_cve_alias") == 1:
                existing["has_cve_alias"] = 1

            if record.get("has_ghsa_alias") == 1:
                existing["has_ghsa_alias"] = 1

    return list(lookup.values())


def limit_records(records, max_count):
    """
    Limit number of vulnerable records.

    Args:
        records (list): Vulnerable package records.
        max_count (int): Maximum number to retain.

    Returns:
        list: Limited records.
    """
    if max_count is None or max_count <= 0:
        return records

    return records[:max_count]


def save_csv(records):
    """Save records to CSV."""
    if not records:
        print("No records to save.")
        return

    fieldnames = [
        "package_name",
        "label",
        "vulnerability_count",
        "osv_id",
        "aliases",
        "has_cve_alias",
        "has_ghsa_alias",
        "published",
        "modified",
        "severity",
        "source",
        "category",
        "confidence"
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow({
                field: record.get(field, "")
                for field in fieldnames
            })


def save_json(records):
    """Save records to JSON."""
    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=4)


# -----------------------------
# Main Logic
# -----------------------------

def main():
    ensure_directories()

    print("=" * 60)
    print("OSV KNOWN VULNERABLE NPM COLLECTOR STARTED")
    print("=" * 60)
    print(f"Max vulnerable packages: {MAX_VULNERABLE_PACKAGES}")

    start_time = time.time()

    zip_bytes = load_osv_zip()
    records = parse_osv_zip(zip_bytes)

    print(f"Unique vulnerable npm packages found: {len(records)}")

    records = limit_records(records, MAX_VULNERABLE_PACKAGES)

    save_csv(records)
    save_json(records)

    elapsed = time.time() - start_time

    print("=" * 60)
    print("OSV KNOWN VULNERABLE NPM COLLECTOR COMPLETE")
    print(f"Saved vulnerable seed packages: {len(records)}")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
