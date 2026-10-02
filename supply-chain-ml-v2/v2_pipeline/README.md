# v2 collection pipeline (matched, randomly sampled)

Put this `v2/` folder in your project root (next to `scripts/`, `data/`, `.venv`). Run everything with the venv active.
Scripts 06 and 07 find your `scripts/04_feature_engineering.py` and `scripts/05_train_models.py` automatically (or pass `--fe` / `--tm`).
**Not tested against the live npm/OSV endpoints** (sandbox has no access). Offline checks done: trimmed metadata gives identical
features to full metadata on all 2,999 of your records; stages 05-07 run end to end on test data.

    python v2/v2_01_osv_index.py                 # OSV advisories -> osv_index.csv (+ snapshot date: cite it)
    python v2/v2_02_names.py                     # full registry name list (or --names-file from all-the-package-names)
    python v2/v2_03_sample.py --n-pos 1500 --pool 15000 [--supplement-file popular.txt]
    python v2/v2_04_fetch.py                     # metadata + downloads; resumable, re-run until nothing left
    python v2/v2_05_match.py --ratio 2           # match on download bin x age bin; read match_report.csv
    python v2/v2_06_features.py                  # 80 features, same extractor as before
    python v2/v2_07_train.py                     # CV + PR-AUC + P@k + temporal hold-out + single-feature AUC
    python v2/v2_07_train.py --drop-downloads    # signal beyond popularity (the matching variable)

## Design decisions to state in the paper
- Positives: random sample of npm packages with >=1 non-malware OSV advisory (all IDs kept); MAL- packages excluded from both classes.
- Controls: uniform random from the full registry, excluding anything in OSV. Unknown vulnerabilities remain (label noise) - say so.
- Matching: 7 download bins x 5 age bins; shortfall per cell is reported, never hidden.
- Popular-package supplement: a uniform sample has almost no high-download packages; if you use `--supplement-file`, disclose it.
- SMOTE off by default (matched set ~1:2); `--smote` reproduces the old setup for comparison.
- Limitation: features are measured today (after disclosure); package age and latest version are post-hoc. Temporal hold-out is a partial check only.
