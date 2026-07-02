"""
04_feature_engineering.py

Purpose:
Create an enhanced machine-learning-ready feature matrix from npm metadata,
OSV vulnerability labels, and npm download statistics.

Inputs:
data/raw/npm_metadata_raw.json
data/interim/npm_osv_enriched.json
data/interim/npm_downloads_enriched.json

Output:
data/processed/npm_features.csv
"""

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

METADATA_FILE = BASE_DIR / "data" / "raw" / "npm_metadata_raw.json"
OSV_FILE = BASE_DIR / "data" / "interim" / "npm_osv_enriched.json"
DOWNLOADS_FILE = BASE_DIR / "data" / "interim" / "npm_downloads_enriched.json"

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "npm_features.csv"

RECENT_UPDATE_DAYS = 30
VERY_NEW_PACKAGE_DAYS = 30
STALE_PACKAGE_DAYS = 730


def ensure_directories():
    (BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_datetime(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def safe_len(value):
    if value is None:
        return 0
    if isinstance(value, (list, dict, str)):
        return len(value)
    return 0


def has_value(value):
    if value is None:
        return 0
    if isinstance(value, str):
        return 1 if value.strip() else 0
    if isinstance(value, (list, dict)):
        return 1 if len(value) > 0 else 0
    return 1


def shannon_entropy(text):
    if not text:
        return 0.0
    probabilities = [text.count(char) / len(text) for char in set(text)]
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


def parse_semver(version):
    if not version:
        return 0, 0, 0

    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(version))
    if not match:
        return 0, 0, 0

    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def get_latest_version_data(metadata):
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
    lookup = {}
    for record in records:
        key = record.get(key_name)
        if key:
            lookup[key] = record
    return lookup


def extract_features(metadata, osv_lookup, downloads_lookup):
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

    total_dependency_count = (
        dependency_count
        + dev_dependency_count
        + optional_dependency_count
        + peer_dependency_count
    )

    maintainer_count = safe_len(maintainers)

    dependency_density = dependency_count / (version_count + 1)
    total_dependency_density = total_dependency_count / (version_count + 1)
    update_frequency = version_count / (package_age_days + 1)
    dependency_growth_proxy = dependency_count / (package_age_days + 1)
    staleness_ratio = days_since_last_update / (package_age_days + 1)

    maintainer_per_version_ratio = maintainer_count / (version_count + 1)
    maintainer_to_dependency_ratio = maintainer_count / (total_dependency_count + 1)
    maintainer_density = maintainer_count / (dependency_count + 1)

    no_maintainer_flag = 1 if maintainer_count == 0 else 0
    single_maintainer_flag = 1 if maintainer_count == 1 else 0
    low_maintainer_flag = 1 if maintainer_count <= 1 else 0

    rapid_release_flag = 1 if update_frequency > 0.05 else 0
    version_burst_flag = 1 if version_count > 50 and package_age_days < 365 else 0
    version_major_jump_flag = 1 if latest_major >= 10 else 0

    is_scoped_package = 1 if name.startswith("@") else 0
    scope_length = len(name.split("/")[0]) if is_scoped_package and "/" in name else 0
    package_name_token_count = len(re.split(r"[-_.\/]", name)) if name else 0
    contains_dash = 1 if "-" in name else 0
    contains_dot = 1 if "." in name else 0
    contains_digit = 1 if any(char.isdigit() for char in name) else 0
    name_length = len(name)
    name_entropy_score = shannon_entropy(name)

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
    short_name_flag = 1 if name_length <= 3 else 0
    deprecated_flag = 1 if "deprecated" in latest_data else 0

    script_count = safe_len(scripts)
    has_scripts = 1 if script_count > 0 else 0
    has_preinstall_script = 1 if "preinstall" in scripts else 0
    has_install_script = 1 if "install" in scripts else 0
    has_postinstall_script = 1 if "postinstall" in scripts else 0
    has_prepare_script = 1 if "prepare" in scripts else 0
    has_prepublish_script = 1 if "prepublish" in scripts else 0

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

    recent_update_flag = 1 if days_since_last_update <= RECENT_UPDATE_DAYS else 0
    very_new_package_flag = 1 if package_age_days <= VERY_NEW_PACKAGE_DAYS else 0
    stale_abandoned_flag = 1 if days_since_last_update >= STALE_PACKAGE_DAYS else 0

    downloads_record = downloads_lookup.get(name, {})
    weekly_downloads = int(downloads_record.get("weekly_downloads", 0) or 0)
    monthly_downloads = int(downloads_record.get("monthly_downloads", 0) or 0)

    log_weekly_downloads = math.log1p(weekly_downloads)
    log_monthly_downloads = math.log1p(monthly_downloads)

    osv_record = osv_lookup.get(name, {})
    label = int(osv_record.get("label", 0))
    vulnerability_count = int(osv_record.get("vulnerability_count", 0) or 0)

    aliases = osv_record.get("aliases", []) or []
    severity = osv_record.get("severity", []) or []

    has_cve_alias = 1 if any(str(alias).startswith("CVE-") for alias in aliases) else 0
    has_ghsa_alias = 1 if any(str(alias).startswith("GHSA-") for alias in aliases) else 0
    severity_count = safe_len(severity)

    return {
        "package_name": name,

        "label": label,
        "vulnerability_count": vulnerability_count,
        "has_cve_alias": has_cve_alias,
        "has_ghsa_alias": has_ghsa_alias,
        "severity_count": severity_count,

        "version_count": version_count,
        "latest_version_major": latest_major,
        "latest_version_minor": latest_minor,
        "latest_version_patch": latest_patch,
        "version_major_jump_flag": version_major_jump_flag,

        "dependency_count": dependency_count,
        "dev_dependency_count": dev_dependency_count,
        "optional_dependency_count": optional_dependency_count,
        "peer_dependency_count": peer_dependency_count,
        "total_dependency_count": total_dependency_count,
        "dependency_density": dependency_density,
        "total_dependency_density": total_dependency_density,
        "dependency_growth_proxy": dependency_growth_proxy,

        "package_age_days": package_age_days,
        "days_since_last_update": days_since_last_update,
        "update_frequency": update_frequency,
        "staleness_ratio": staleness_ratio,
        "recent_update_flag": recent_update_flag,
        "very_new_package_flag": very_new_package_flag,
        "stale_abandoned_flag": stale_abandoned_flag,

        "maintainer_count": maintainer_count,
        "no_maintainer_flag": no_maintainer_flag,
        "single_maintainer_flag": single_maintainer_flag,
        "low_maintainer_flag": low_maintainer_flag,
        "maintainer_per_version_ratio": maintainer_per_version_ratio,
        "maintainer_to_dependency_ratio": maintainer_to_dependency_ratio,
        "maintainer_density": maintainer_density,

        "is_scoped_package": is_scoped_package,
        "scope_length": scope_length,
        "package_name_token_count": package_name_token_count,
        "contains_dash": contains_dash,
        "contains_dot": contains_dot,
        "contains_digit": contains_digit,
        "name_length": name_length,
        "name_entropy_score": name_entropy_score,

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
        "short_name_flag": short_name_flag,
        "deprecated_flag": deprecated_flag,

        "script_count": script_count,
        "has_scripts": has_scripts,
        "has_preinstall_script": has_preinstall_script,
        "has_install_script": has_install_script,
        "has_postinstall_script": has_postinstall_script,
        "has_prepare_script": has_prepare_script,
        "has_prepublish_script": has_prepublish_script,
        "has_lifecycle_script": has_lifecycle_script,
        "suspicious_script_flag": suspicious_script_flag,

        "weekly_downloads": weekly_downloads,
        "monthly_downloads": monthly_downloads,
        "log_weekly_downloads": log_weekly_downloads,
        "log_monthly_downloads": log_monthly_downloads
    }


def clean_dataframe(df):
    df = df.drop_duplicates(subset=["package_name"])
    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    text_columns = df.select_dtypes(include=["object"]).columns
    df[text_columns] = df[text_columns].fillna("")

    return df


def main():
    ensure_directories()

    print("=" * 60)
    print("ENHANCED FEATURE ENGINEERING STARTED")
    print("=" * 60)

    metadata_records = load_json(METADATA_FILE)
    osv_records = load_json(OSV_FILE)
    downloads_records = load_json(DOWNLOADS_FILE)

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

    print("=" * 60)
    print("ENHANCED FEATURE ENGINEERING COMPLETE")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Feature count excluding package_name and label: {len(df.columns) - 2}")
    print(f"Vulnerable packages: {int(df['label'].sum())}")
    print(f"Benign packages: {int(len(df) - df['label'].sum())}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
