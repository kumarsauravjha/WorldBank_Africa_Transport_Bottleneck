# Breaking Africa’s Transport Bottlenecks
_A World Bank project analyzing transport costs, anomalies, and opportunities using the GTCDIT dataset._

This repository contains Python scripts and reports used to process, analyze, and visualize international transport costs, with a special focus on African exporters.  
The workflow builds on the **Global Transport Costs Dataset for International Trade (GTCDIT)** (2016–2021), and adds derived features, anomaly detection, classification, and targeted Africa-specific studies.

---

## Dataset
- **Input (WB provided):** `imputed_full_matrix_at_centroid.csv`
- **Enriched working file:** `wb_data_all_fields_final.csv` (with added geo, trade-type, and income group fields)

---

## Python Scripts

### 1. `feature_engineering.py`
Adds derived fields to the raw WB dataset, including continent, region, trade type, landlocked/coastal flags, distance bins, and income groups. Produces the enriched dataset for all downstream analysis.

### 2. `data_cleaning.py`
Filters invalid records (zero/negative flows or costs), applies **IsolationForest** + IQR rules for global anomaly detection, and visualizes overall distributions. Focused on cleaning the dataset and surfacing global outliers.

### 3. `anomaly.py`
Runs targeted anomaly detection on **Africa + overseas trades**, using IsolationForest and log–log scatter plots to highlight implausible distance–cost relationships and commodity–mode mismatches.

### 4. `Africa Income Group Analysis.py`
Breaks down African exports by **income group**, analyzing mode shares, top commodities, costs, destinations, and country counts. Produces comparative plots and 2×2 dashboards.

### 5. `african_exports_analysis.py`
End-to-end African exports study: descriptive stats, mode/region breakdowns, **XGBoost cost model** with SHAP explainability, and anomaly scans (landlocked vs coastal, high-cost exporters, RoRo air anomalies). Saves figures and CSVs.

### 6. `cost_classification_script.py`
Classifies routes into **Low/Moderate/High cost categories** using three methods: percentile bins, K-means clustering, and rule-based thresholds. Outputs histograms, stacked bars, and cost–distance scatterplots.

### 7. `create_africa_avg_cost_combined.py`
Utility script that filters the dataset for African origins, computes **average unit costs per route/commodity/mode**, saves results to CSV, and provides summary stats for diagnostics/web integration.

---

## Outputs
- **Figures:** Charts on anomalies, mode distribution, export costs, SHAP feature importance, etc.
- **CSV tables:** Cleaned subsets, cost classifications, Africa-only averages, mode-shift/corridor opportunities.

---

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
