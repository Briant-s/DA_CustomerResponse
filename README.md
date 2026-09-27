# Customer Campaign Response Analytics

An end-to-end data analytics project that explores a retail marketing dataset and builds a
machine learning model to predict whether a customer will respond to a marketing campaign.

The full workflow — from raw data to a trained classifier — lives in a single, well-structured
Jupyter notebook: [`final_group8.ipynb`](./final_group8.ipynb).

## Overview

Marketing teams spend budget on campaigns without knowing who is likely to convert. This project
analyzes historical customer data (demographics, spending habits, and past campaign outcomes) to:

- Understand what drives a customer to accept a campaign offer.
- Test hypotheses about customer behavior with statistical rigor.
- Train a model that predicts the `Response` target so campaigns can be targeted more effectively.

The dataset contains **2,240 customers** across **28 columns**, sourced from
[`marketing_data.csv`](./marketing_data.csv).

## Workflow

The notebook is organized into six sequential stages:

1. **Initial Data Overview** — Inspect structure, data types, missing values, and unique
   categorical values.
2. **Data Cleansing** — Handle missing income values, fix inconsistent categories, and correct
   data types.
3. **Exploratory Data Analysis (EDA)** — Descriptive statistics, visualizations, hypothesis
   testing, and correlation analysis.
4. **Data Preparation** — Train/test split, outlier capping (IQR), feature engineering, encoding
   (ordinal + one-hot), multicollinearity checks (VIF), and scaling.
5. **Modelling** — Train a Logistic Regression classifier on the prepared features.
6. **Evaluation** — Confusion matrix, classification report, ROC curve, precision–recall curve,
   and feature importance.

> [!NOTE]
> To prevent data leakage, all preprocessing steps (outlier boundaries, encoders, scaler) are
> **fit on the training set only** and then applied to the test set.

## Dataset

Key fields from the data dictionary:

| Field | Description |
|-------|-------------|
| `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Country` | Customer demographics |
| `Kidhome`, `Teenhome` | Number of children / teenagers at home |
| `Recency` | Days since last purchase |
| `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` | Spending per category (last 2 years) |
| `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumDealsPurchases` | Purchase channels |
| `NumWebVisitsMonth` | Website visits in the last month |
| `AcceptedCmp1`–`AcceptedCmp5` | Whether prior campaigns were accepted |
| `Response` | **Target** — accepted the last campaign (1) or not (0) |

## Tech Stack

- **Python 3.13**
- **pandas** / **numpy** — data manipulation
- **matplotlib** / **seaborn** — visualization
- **scikit-learn** — modeling, preprocessing, and evaluation
- **statsmodels** / **pingouin** — statistical analysis (VIF, hypothesis testing)
- **uv** — dependency and environment management

## Getting Started

### Prerequisites

- [Python 3.13+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/) for dependency management

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd Data_Analytics

# Create the environment and install dependencies
uv sync
```

### Run the analysis

Launch Jupyter and open the notebook:

```bash
uv run jupyter notebook final_group8.ipynb
```

Then run the cells top to bottom to reproduce the full analysis and model.

> [!TIP]
> Prefer VS Code? Open `final_group8.ipynb` and select the `.venv` interpreter created by
> `uv sync` as the notebook kernel.

## Results

The trained Logistic Regression model is evaluated with multiple metrics, and its outputs are
saved as images in the project root:

| Artifact | Description |
|----------|-------------|
| [`correlation_heatmaps.png`](./correlation_heatmaps.png) | Feature correlation matrix |
| [`feature_importance.png`](./feature_importance.png) | Logistic Regression coefficients |
| [`time_trend_analysis.png`](./time_trend_analysis.png) | Enrollment / response trends over time |

Evaluation covers the confusion matrix, precision/recall/F1 (classification report), ROC–AUC,
and average precision — giving a balanced view of performance on an imbalanced target.

## Project Structure

```
Data_Analytics/
├── final_group8.ipynb        # Main analysis notebook (EDA → model → evaluation)
├── marketing_data.csv        # Raw dataset (2,240 customers, 28 columns)
├── correlation_heatmaps.png  # Generated: correlation analysis
├── feature_importance.png    # Generated: model coefficients
├── time_trend_analysis.png   # Generated: time-based trends
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Locked dependency versions
└── main.py                   # Package entry point
```
