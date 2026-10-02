"""
04_feature_engineering.py

Purpose:
Create an enhanced machine-learning-ready feature matrix from npm background
metadata, OSV vulnerable seed metadata, OSV vulnerability labels, and npm
download statistics.

Inputs:
data/raw/npm_metadata_raw.json
data/raw/npm_vulnerable_metadata_raw.json
data/interim/npm_osv_enriched.json
data/interim/npm_downloads_enriched.json

Output:
data/processed/npm_features.csv

Important:
The final dataset is constructed from:
1. Background npm package sample
2. OSV-confirmed vulnerable package seed metadata

The vulnerable seed packages contain _seed_label = 1 and are therefore labelled
as vulnerable even if they were not included in the original OSV enrichment file.
"""

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

METADATA_FILE = BASE_DIR / "data" / "raw" / "npm_metadata_raw.json"
VULNERABLE_METADATA_FILE = BASE_DIR / "data" / "raw" / "npm_vulnerable_metadata_raw.json"

OSV_FILE = BASE_DIR / "data" / "interim" / "npm_osv_enriched.json"
DOWNLOADS_FILE = BASE_DIR / "data" / "interim" / "npm_downloads_enriched.json"

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "npm_features.csv"

RECENT_UPDATE_DAYS = 30
VERY_NEW_PACKAGE_DAYS = 30
STALE_PACKAGE_DAYS = 730


# -----------------------------
# Utility Functions
# -----------------------------

def ensure_directories():
    """Create output directory if it does not exist."""
    (BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)


def load_json(path, required=True):
    """
    Load a JSON file.

    Args:
        path (Path): File path.
        required (bool): If True, raise error when file is missing.

    Returns:
        list | dict
    """
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise FileNotFoundError(f"Required file not found or empty: {path}")
        return []

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_datetime(value):
    """Convert npm timestamp string into timezone-aware datetime."""
    if not value:
        return None

    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def safe_len(value):
    """Safely compute length of list, dict, or string values."""
    if value is None:
        return 0

    if isinstance(value, (list, dict, str)):
        return len(value)

    return 0


def has_value(value):
    """Return 1 if a metadata field contains a meaningful value."""
    if value is None:
        return 0

    if isinstance(value, str):
        return 1 if value.strip() else 0

    if isinstance(value, (list, dict)):
        return 1 if len(value) > 0 else 0

    return 1


def shannon_entropy(text):
    """
    Compute Shannon entropy of a string.

    Higher entropy may indicate randomised, obfuscated, or automatically
    generated package names.
    """
    if not text:
        return 0.0

    probabilities = [
        text.count(char) / len(text)
        for char in set(text)
    ]

    return -sum(
        probability * math.log2(probability)
        for probability in probabilities
        if probability > 0
    )


def parse_semver(version):
    """
    Parse semantic version into major, minor, and patch values.

    Non-standard versions are returned as 0,0,0.
    """
    if not version:
        return 0, 0, 0

    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(version))

    if not match:
        return 0, 0, 0

    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3))
    )


def get_latest_version_data(metadata):
    """
    Retrieve latest version data from npm metadata.

    Returns:
        tuple: latest_version, latest_version_metadata
    """
    versions = metadata.get("versions", {}) or {}
    latest_version = metadata.get("dist-tags", {}).get("latest")

    if latest_version and latest_version in versions:
        return latest_version, versions.get(latest_version, {}) or {}

    if versions:
        try:
            latest_version = list(versions.keys())[-1]
            return latest_version, list(versions.values())[-1] or {}
        except Exception:
            return "", {}

    return "", {}


def create_lookup(records, key_name="package_name"):
    """Create dictionary lookup from a list of records."""
    lookup = {}

    for record in records:
        key = record.get(key_name)

        if key:
            lookup[key] = record

    return lookup


def merge_and_deduplicate_metadata(background_records, vulnerable_records):
    """
    Merge background and vulnerable metadata records.

    If the same package appears in both datasets, vulnerable seed metadata is
    retained because it contains explicit provenance and _seed_label = 1.
    """
    merged = {}

    for record in background_records:
        name = record.get("name")
        if name:
            merged[name] = record

    for record in vulnerable_records:
        name = record.get("name")
        if name:
            merged[name] = record

    return list(merged.values())


# -----------------------------
# Feature Extraction
# -----------------------------

def extract_features(metadata, osv_lookup, downloads_lookup):
    """
    Extract engineered features from one npm metadata record.

    Args:
        metadata (dict): npm metadata.
        osv_lookup (dict): OSV vulnerability label lookup.
        downloads_lookup (dict): npm download statistics lookup.

    Returns:
        dict: Feature row.
    """
    now = datetime.now(timezone.utc)

    name = metadata.get("name", "") or ""
    description = metadata.get("description", "") or ""
    readme = metadata.get("readme", "") or ""

    versions = metadata.get("versions", {}) or {}
    time_info = metadata.get("time", {}) or {}
    maintainers = metadata.get("maintainers", []) or []
    keywords = metadata.get("keywords", []) or []

    latest_version, latest_data = get_latest_version_data(metadata)

    dependencies = latest_data.get("dependencies", {}) or {}
    dev_dependencies = latest_data.get("devDependencies", {}) or {}
    optional_dependencies = latest_data.get("optionalDependencies", {}) or {}
    peer_dependencies = latest_data.get("peerDependencies", {}) or {}
    bundled_dependencies = latest_data.get("bundledDependencies", {}) or {}
    scripts = latest_data.get("scripts", {}) or {}

    repository = metadata.get("repository") or latest_data.get("repository")
    homepage = metadata.get("homepage") or latest_data.get("homepage")
    bugs = metadata.get("bugs") or latest_data.get("bugs")
    license_info = metadata.get("license") or latest_data.get("license")
    author = metadata.get("author") or latest_data.get("author")

    created_date = parse_datetime(time_info.get("created"))
    modified_date = parse_datetime(time_info.get("modified"))

    package_age_days = max((now - created_date).days, 0) if created_date else 0
    days_since_last_update = max((now - modified_date).days, 0) if modified_date else 0

    version_count = safe_len(versions)
    latest_major, latest_minor, latest_patch = parse_semver(latest_version)

    dependency_count = safe_len(dependencies)
    dev_dependency_count = safe_len(dev_dependencies)
    optional_dependency_count = safe_len(optional_dependencies)
    peer_dependency_count = safe_len(peer_dependencies)
    bundled_dependency_count = safe_len(bundled_dependencies)

    total_dependency_count = (
        dependency_count
        + dev_dependency_count
        + optional_dependency_count
        + peer_dependency_count
        + bundled_dependency_count
    )

    maintainer_count = safe_len(maintainers)

    # -----------------------------
    # Structural and lifecycle features
    # -----------------------------

    dependency_density = dependency_count / (version_count + 1)
    total_dependency_density = total_dependency_count / (version_count + 1)
    update_frequency = version_count / (package_age_days + 1)
    dependency_growth_proxy = dependency_count / (package_age_days + 1)
    staleness_ratio = days_since_last_update / (package_age_days + 1)

    dev_dependency_ratio = dev_dependency_count / (total_dependency_count + 1)
    peer_dependency_ratio = peer_dependency_count / (total_dependency_count + 1)
    optional_dependency_ratio = optional_dependency_count / (total_dependency_count + 1)

    # -----------------------------
    # Maintainer trust features
    # -----------------------------

    maintainer_per_version_ratio = maintainer_count / (version_count + 1)
    maintainer_to_dependency_ratio = maintainer_count / (total_dependency_count + 1)
    maintainer_density = maintainer_count / (dependency_count + 1)

    no_maintainer_flag = 1 if maintainer_count == 0 else 0
    single_maintainer_flag = 1 if maintainer_count == 1 else 0
    low_maintainer_flag = 1 if maintainer_count <= 1 else 0

    # -----------------------------
    # Version behaviour features
    # -----------------------------

    rapid_release_flag = 1 if update_frequency > 0.05 else 0
    version_burst_flag = 1 if version_count > 50 and package_age_days < 365 else 0
    version_major_jump_flag = 1 if latest_major >= 10 else 0

    # -----------------------------
    # Package identity and naming features
    # -----------------------------

    is_scoped_package = 1 if name.startswith("@") else 0
    scope_length = len(name.split("/")[0]) if is_scoped_package and "/" in name else 0
    package_name_token_count = len(re.split(r"[-_.\/]", name)) if name else 0

    contains_dash = 1 if "-" in name else 0
    contains_dot = 1 if "." in name else 0
    contains_digit = 1 if any(char.isdigit() for char in name) else 0
    contains_underscore = 1 if "_" in name else 0

    name_length = len(name)
    name_entropy_score = shannon_entropy(name)

    # -----------------------------
    # Repository transparency features
    # -----------------------------

    has_repository = has_value(repository)
    has_github_repository = 1 if "github.com" in str(repository).lower() else 0
    has_homepage = has_value(homepage)
    has_bugs_url = has_value(bugs)
    has_license = has_value(license_info)
    has_author = has_value(author)
    has_keywords = 1 if len(keywords) > 0 else 0
    keyword_count = safe_len(keywords)

    description_length = len(description)
    readme_length = len(readme)

    description_missing_flag = 1 if description_length == 0 else 0
    readme_missing_flag = 1 if readme_length == 0 else 0
    short_name_flag = 1 if name_length <= 3 else 0
    deprecated_flag = 1 if "deprecated" in latest_data else 0

    transparency_score = (
        has_repository
        + has_homepage
        + has_bugs_url
        + has_license
        + has_author
        + has_keywords
    )

    # -----------------------------
    # npm script execution features
    # -----------------------------

    script_count = safe_len(scripts)
    has_scripts = 1 if script_count > 0 else 0

    has_preinstall_script = 1 if "preinstall" in scripts else 0
    has_install_script = 1 if "install" in scripts else 0
    has_postinstall_script = 1 if "postinstall" in scripts else 0
    has_prepare_script = 1 if "prepare" in scripts else 0
    has_prepublish_script = 1 if "prepublish" in scripts else 0
    has_prepublish_only_script = 1 if "prepublishOnly" in scripts else 0

    has_lifecycle_script = 1 if any(
        key in scripts
        for key in [
            "preinstall",
            "install",
            "postinstall",
            "prepare",
            "prepublish",
            "prepublishOnly"
        ]
    ) else 0

    suspicious_script_flag = has_lifecycle_script

    script_text = " ".join(str(value).lower() for value in scripts.values())

    script_uses_curl = 1 if "curl" in script_text else 0
    script_uses_wget = 1 if "wget" in script_text else 0
    script_uses_eval = 1 if "eval" in script_text else 0
    script_uses_base64 = 1 if "base64" in script_text else 0
    script_uses_child_process = 1 if "child_process" in script_text else 0

    suspicious_script_keyword_count = (
        script_uses_curl
        + script_uses_wget
        + script_uses_eval
        + script_uses_base64
        + script_uses_child_process
    )

    # -----------------------------
    # Temporal risk indicators
    # -----------------------------

    recent_update_flag = 1 if days_since_last_update <= RECENT_UPDATE_DAYS else 0
    very_new_package_flag = 1 if package_age_days <= VERY_NEW_PACKAGE_DAYS else 0
    stale_abandoned_flag = 1 if days_since_last_update >= STALE_PACKAGE_DAYS else 0

    # -----------------------------
    # Download popularity features
    # -----------------------------

    downloads_record = downloads_lookup.get(name, {})

    weekly_downloads = int(downloads_record.get("weekly_downloads", 0) or 0)
    monthly_downloads = int(downloads_record.get("monthly_downloads", 0) or 0)

    log_weekly_downloads = math.log1p(weekly_downloads)
    log_monthly_downloads = math.log1p(monthly_downloads)

    download_ratio_week_month = weekly_downloads / (monthly_downloads + 1)

    high_download_flag = 1 if monthly_downloads >= 100000 else 0
    low_download_flag = 1 if monthly_downloads <= 100 else 0

    # -----------------------------
    # OSV and seed label features
    # -----------------------------

    osv_record = osv_lookup.get(name, {})

    seed_label = int(metadata.get("_seed_label", 0) or 0)
    osv_label = int(osv_record.get("label", 0) or 0)

    label = 1 if seed_label == 1 or osv_label == 1 else 0

    vulnerability_count = int(
        metadata.get("_seed_vulnerability_count", 0)
        or osv_record.get("vulnerability_count", 0)
        or 0
    )

    aliases = osv_record.get("aliases", []) or []
    severity = osv_record.get("severity", []) or []

    seed_aliases = metadata.get("_seed_aliases", "")
    if seed_aliases:
        aliases = aliases + str(seed_aliases).split(";")

    has_cve_alias = 1 if any(str(alias).startswith("CVE-") for alias in aliases) else 0
    has_ghsa_alias = 1 if any(str(alias).startswith("GHSA-") for alias in aliases) else 0
    severity_count = safe_len(severity)

    # -----------------------------
    # Composite risk proxy features
    # -----------------------------

    metadata_risk_score = (
        description_missing_flag
        + readme_missing_flag
        + short_name_flag
        + (1 - has_repository)
        + (1 - has_license)
        + deprecated_flag
    )

    dependency_risk_score = (
        low_maintainer_flag
        + stale_abandoned_flag
        + rapid_release_flag
        + version_burst_flag
        + suspicious_script_flag
    )

    return {
        "package_name": name,

        # Target and vulnerability features
        "label": label,
        "vulnerability_count": vulnerability_count,
        "has_cve_alias": has_cve_alias,
        "has_ghsa_alias": has_ghsa_alias,
        "severity_count": severity_count,
        "seed_label": seed_label,
        "osv_label": osv_label,

        # Version and structural features
        "version_count": version_count,
        "latest_version_major": latest_major,
        "latest_version_minor": latest_minor,
        "latest_version_patch": latest_patch,
        "version_major_jump_flag": version_major_jump_flag,

        "dependency_count": dependency_count,
        "dev_dependency_count": dev_dependency_count,
        "optional_dependency_count": optional_dependency_count,
        "peer_dependency_count": peer_dependency_count,
        "bundled_dependency_count": bundled_dependency_count,
        "total_dependency_count": total_dependency_count,
        "dependency_density": dependency_density,
        "total_dependency_density": total_dependency_density,
        "dependency_growth_proxy": dependency_growth_proxy,
        "dev_dependency_ratio": dev_dependency_ratio,
        "peer_dependency_ratio": peer_dependency_ratio,
        "optional_dependency_ratio": optional_dependency_ratio,

        # Temporal features
        "package_age_days": package_age_days,
        "days_since_last_update": days_since_last_update,
        "update_frequency": update_frequency,
        "staleness_ratio": staleness_ratio,
        "recent_update_flag": recent_update_flag,
        "very_new_package_flag": very_new_package_flag,
        "stale_abandoned_flag": stale_abandoned_flag,

        # Maintainer features
        "maintainer_count": maintainer_count,
        "no_maintainer_flag": no_maintainer_flag,
        "single_maintainer_flag": single_maintainer_flag,
        "low_maintainer_flag": low_maintainer_flag,
        "maintainer_per_version_ratio": maintainer_per_version_ratio,
        "maintainer_to_dependency_ratio": maintainer_to_dependency_ratio,
        "maintainer_density": maintainer_density,

        # Naming features
        "is_scoped_package": is_scoped_package,
        "scope_length": scope_length,
        "package_name_token_count": package_name_token_count,
        "contains_dash": contains_dash,
        "contains_dot": contains_dot,
        "contains_digit": contains_digit,
        "contains_underscore": contains_underscore,
        "name_length": name_length,
        "name_entropy_score": name_entropy_score,

        # Transparency and metadata features
        "has_repository": has_repository,
        "has_github_repository": has_github_repository,
        "has_homepage": has_homepage,
        "has_bugs_url": has_bugs_url,
        "has_license": has_license,
        "has_author": has_author,
        "has_keywords": has_keywords,
        "keyword_count": keyword_count,
        "description_length": description_length,
        "readme_length": readme_length,
        "description_missing_flag": description_missing_flag,
        "readme_missing_flag": readme_missing_flag,
        "short_name_flag": short_name_flag,
        "deprecated_flag": deprecated_flag,
        "transparency_score": transparency_score,

        # Script execution features
        "script_count": script_count,
        "has_scripts": has_scripts,
        "has_preinstall_script": has_preinstall_script,
        "has_install_script": has_install_script,
        "has_postinstall_script": has_postinstall_script,
        "has_prepare_script": has_prepare_script,
        "has_prepublish_script": has_prepublish_script,
        "has_prepublish_only_script": has_prepublish_only_script,
        "has_lifecycle_script": has_lifecycle_script,
        "suspicious_script_flag": suspicious_script_flag,
        "script_uses_curl": script_uses_curl,
        "script_uses_wget": script_uses_wget,
        "script_uses_eval": script_uses_eval,
        "script_uses_base64": script_uses_base64,
        "script_uses_child_process": script_uses_child_process,
        "suspicious_script_keyword_count": suspicious_script_keyword_count,

        # Download features
        "weekly_downloads": weekly_downloads,
        "monthly_downloads": monthly_downloads,
        "log_weekly_downloads": log_weekly_downloads,
        "log_monthly_downloads": log_monthly_downloads,
        "download_ratio_week_month": download_ratio_week_month,
        "high_download_flag": high_download_flag,
        "low_download_flag": low_download_flag,

        # Composite proxy features
        "metadata_risk_score": metadata_risk_score,
        "dependency_risk_score": dependency_risk_score
    }


# -----------------------------
# Cleaning
# -----------------------------

def clean_dataframe(df):
    """Clean engineered feature dataframe."""
    df = df.drop_duplicates(subset=["package_name"])
    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    text_columns = df.select_dtypes(include=["object"]).columns
    df[text_columns] = df[text_columns].fillna("")

    return df


# -----------------------------
# Main
# -----------------------------

def main():
    ensure_directories()

    print("=" * 60)
    print("ENHANCED FEATURE ENGINEERING STARTED")
    print("=" * 60)

    background_metadata_records = load_json(METADATA_FILE, required=True)
    vulnerable_metadata_records = load_json(VULNERABLE_METADATA_FILE, required=False)

    osv_records = load_json(OSV_FILE, required=False)
    downloads_records = load_json(DOWNLOADS_FILE, required=False)

    print(f"Background metadata records: {len(background_metadata_records)}")
    print(f"Vulnerable seed metadata records: {len(vulnerable_metadata_records)}")
    print(f"OSV label records: {len(osv_records)}")
    print(f"Download records: {len(downloads_records)}")

    metadata_records = merge_and_deduplicate_metadata(
        background_metadata_records,
        vulnerable_metadata_records
    )

    print(f"Merged unique metadata records: {len(metadata_records)}")

    osv_lookup = create_lookup(osv_records, "package_name")
    downloads_lookup = create_lookup(downloads_records, "package_name")

    feature_rows = []

    for metadata in metadata_records:
        try:
            feature_rows.append(
                extract_features(metadata, osv_lookup, downloads_lookup)
            )
        except Exception as error:
            print(f"Feature extraction failed for {metadata.get('name')}: {error}")

    df = pd.DataFrame(feature_rows)
    df = clean_dataframe(df)

    df.to_csv(OUTPUT_FILE, index=False)

    vulnerable_count = int(df["label"].sum())
    benign_count = int(len(df) - vulnerable_count)

    print("=" * 60)
    print("ENHANCED FEATURE ENGINEERING COMPLETE")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Feature count excluding package_name and label: {len(df.columns) - 2}")
    print(f"Vulnerable packages: {vulnerable_count}")
    print(f"Benign packages: {benign_count}")

    if len(df) > 0:
        print(f"Vulnerable percentage: {(vulnerable_count / len(df)) * 100:.2f}%")

    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
