# SupplyChainDetection

A Machine Learning-Based Framework for Detecting Vulnerabilities and Malicious Patterns in Open-Source Software Dependencies.

---

## Overview

SupplyChainDetection is an MSc Cybersecurity and Digital Forensics research project that investigates software supply chain attacks targeting open-source dependencies and proposes an intelligent machine learning-based detection framework.

The framework collects package metadata from the npm ecosystem, enriches it with vulnerability intelligence from the Open Source Vulnerability (OSV) database and npm download statistics, engineers security-related features, and trains machine learning models to identify vulnerable or potentially malicious packages.

The project is designed to support:

- Software supply chain risk assessment
- Vulnerability detection
- Open-source dependency analysis
- Secure software development practices
- DevSecOps integration

---

## Research Objectives

This project seeks to:

1. Analyse supply chain attack vectors targeting open-source dependencies.
2. Identify and classify vulnerabilities in open-source packages using machine learning techniques.
3. Develop an intelligent detection framework for recognising malicious patterns in software supply chains.
4. Evaluate the effectiveness of machine learning models in detecting vulnerable dependencies.
5. Investigate mitigation strategies including:
   - Software Bill of Materials (SBOM)
   - Vulnerability scanning
   - Enhanced authentication mechanisms
   - Dependency monitoring

---

## Architecture

```text
Package Collection
        ↓
Metadata Enrichment
        ↓
Vulnerability Enrichment
        ↓
Download Statistics Enrichment
        ↓
Feature Engineering
        ↓
Machine Learning Models
        ↓
Prediction and Risk Assessment
```

---

## Project Structure

```text
supply-chain-ml/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── figures/
├── logs/
├── models/
├── notebooks/
├── reports/
├── results/
│
├── scripts/
│   ├── 01_npm_collector.py
│   ├── 02_npm_metadata_enricher.py
│   ├── 03_osv_vulnerability_enricher.py
│   ├── 03b_npm_download_enricher.py
│   ├── 04_feature_engineering.py
│   ├── 05_train_models.py
│   └── 06_generate_figures.py
│
├── requirements.txt
└── README.md
```

---

## Dataset Sources

### npm Registry

https://registry.npmjs.org/

### npm Replication Endpoint

https://replicate.npmjs.com/_all_docs

### npm Download Statistics API

https://api.npmjs.org/downloads/

### Open Source Vulnerability Database (OSV)

https://osv.dev/

---

## Engineered Feature Categories

The framework extracts features from multiple dimensions:

### Package Identity Features

- Name entropy
- Scoped package indicators
- Naming characteristics

### Dependency Features

- Dependency count
- Dependency density
- Dependency growth indicators

### Maintainer Trust Features

- Maintainer count
- Maintainer-to-dependency ratio
- Single-maintainer risk indicators

### Temporal Features

- Package age
- Update frequency
- Staleness indicators

### Script Execution Features

- Install scripts
- Post-install scripts
- Lifecycle scripts

### Popularity Features

- Weekly downloads
- Monthly downloads
- Log-transformed download statistics

### Vulnerability Features

- Vulnerability count
- CVE indicators
- GHSA indicators

---

## Machine Learning Models

The framework evaluates several supervised machine learning algorithms, including:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

Performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Matthews Correlation Coefficient (MCC)

---

## Installation

### Clone Repository

```bash
git clone https://github.com/<username>/SupplyChainDetection.git
cd SupplyChainDetection
```

### Create Virtual Environment

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Collect npm Packages

```bash
python3 scripts/01_npm_collector.py
```

### Step 2: Enrich Package Metadata

```bash
python3 scripts/02_npm_metadata_enricher.py
```

### Step 3: Enrich Vulnerability Information

```bash
python3 scripts/03_osv_vulnerability_enricher.py
```

### Step 4: Collect Download Statistics

```bash
python3 scripts/03b_npm_download_enricher.py
```

### Step 5: Generate Features

```bash
python3 scripts/04_feature_engineering.py
```

### Step 6: Train Models

```bash
python3 scripts/05_train_models.py
```

### Step 7: Generate Figures and Reports

```bash
python3 scripts/06_generate_figures.py
```

---

## Research Outputs

The framework produces:

- Feature-engineered datasets
- Trained machine learning models
- Performance evaluation reports
- ROC curves
- Confusion matrices
- Feature importance visualisations
- Software supply chain risk assessments

---

## Ethical Considerations

This research uses publicly available data obtained from:

- npm
- OSV
- Public vulnerability databases

No personal or confidential information is collected or processed.

The framework is intended solely for:

- Defensive cybersecurity research
- Vulnerability analysis
- Software supply chain security enhancement

---

## Citation

If you use this work, please cite:

> Yamoah, K. A. (2026). Securing Open-Source Dependencies in International Software Projects: A Machine Learning Approach. MSc Thesis, Wisconsin International University College, Ghana.

---

## Author

**Kwesi Annan Yamoah**

MSc Cybersecurity and Digital Forensics  
Wisconsin International University College, Ghana

---

## License

This project is released under the MIT License.
