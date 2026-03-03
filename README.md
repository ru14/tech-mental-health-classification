# Mental Health in Tech – CRISP-DM Analysis

Predicting mental health treatment-seeking behavior in the tech industry using the CRISP-DM framework and machine learning classification models.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Dataset Description](#dataset-description)
4. [CRISP-DM Framework](#crisp-dm-framework)
5. [Technical Walkthrough](#technical-walkthrough)
6. [Results](#results)
7. [Summary of Findings](#summary-of-findings)
8. [Business Recommendations](#business-recommendations)
9. [Project Structure](#project-structure)
10. [Setup & Installation](#setup--installation)

---

## Project Overview

This project analyzes the [OSMI Mental Health in Tech Survey](https://osmihelp.org/research) dataset to understand what workplace and demographic factors predict whether a tech employee will seek mental health treatment. We apply the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology end-to-end, from business understanding through deployment recommendations.

Mental health issues are prevalent in the tech industry, yet many employees avoid seeking treatment due to stigma, lack of awareness, or unsupportive workplace environments. By building a binary classification model, this project helps organizations identify at-risk groups and design more effective wellness programs.

---

## Business Problem

> **Can we predict whether a tech employee will seek mental health treatment based on workplace and demographic factors?**

Understanding the key drivers of treatment-seeking behavior can help organizations:
- Design better mental health benefit programs
- Reduce stigma through targeted awareness campaigns
- Support employees who may be at risk of untreated mental health conditions

### Success Criteria

- Achieve **F1-score ≥ 0.75** on the classification task
- Identify at least **3 actionable workplace factors** influencing treatment-seeking
- Deliver non-technical business recommendations

---

## Dataset Description

The dataset (`data/survey.csv`) is sourced from the OSMI Mental Health in Tech Survey and contains responses from **1,259 tech industry employees**. Key columns include:

| Column | Description |
|--------|-------------|
| `Age` | Respondent age |
| `Gender` | Respondent gender |
| `Country` | Country of employment |
| `self_employed` | Whether the respondent is self-employed |
| `family_history` | Family history of mental illness (Yes/No) |
| `treatment` | **Target** – sought mental health treatment (Yes/No) |
| `work_interfere` | How often mental health interferes with work |
| `no_employees` | Company size |
| `remote_work` | Whether the respondent works remotely |
| `tech_company` | Whether the employer is a tech company |
| `benefits` | Whether employer provides mental health benefits |
| `care_options` | Awareness of mental health care options |
| `anonymity` | Whether anonymity is protected when seeking help |
| `leave` | Ease of taking medical leave for mental health |
| `mental_health_consequence` | Perceived negative consequences for discussing mental health |
| `phys_health_consequence` | Perceived negative consequences for discussing physical health |
| `coworkers` | Comfort discussing mental health with coworkers |
| `supervisor` | Comfort discussing mental health with supervisor |
| `mental_vs_physical` | Whether employer takes mental health as seriously as physical |
| `obs_consequence` | Whether the respondent has observed negative consequences for coworkers |

---

## CRISP-DM Framework

![CRISP-DM Diagram](images/crisp_dm_diagram.png)

This project follows all six CRISP-DM phases:

1. **Business Understanding** – Define the prediction problem and success criteria
2. **Data Understanding** – Explore the dataset, check distributions and missing values
3. **Data Preparation** – Clean data, handle outliers, engineer features, encode variables
4. **Modeling** – Train Logistic Regression (baseline) and Random Forest with GridSearchCV
5. **Evaluation** – Compare models using Accuracy, Precision, Recall, and F1-score
6. **Deployment & Recommendations** – Derive actionable business insights

---

## Technical Walkthrough

### 1. Data Understanding
- Load and inspect the raw survey data (shape, dtypes, missing values)
- Analyze the **target variable** (`treatment`) distribution for class balance
- Generate descriptive statistics on all numeric and categorical features

### 2. Data Preparation

| Step | Action |
|------|--------|
| Missing Values | Categorical columns filled with `'Unknown'`; numeric columns left as-is |
| Age Filtering | Removed unrealistic ages (kept 15 < Age < 100) |
| Deduplication | Dropped exact duplicate rows |
| Gender Standardization | Normalized free-text gender entries to `Male`, `Female`, or `Other` |
| Feature Engineering | Created `age_group` bins: `16-25`, `26-35`, `36-45`, `46-60`, `60+` |
| Outlier Analysis | Used IQR method on `Age`; extreme values documented but retained after filtering |
| Encoding | Label-encoded all categorical features for model input |
| Train/Test Split | 80% training / 20% test split, stratified by target |

### 3. Modeling

Two classifiers were trained and evaluated:

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline linear classifier (`max_iter=1000`) |
| **Random Forest** | Ensemble model tuned with `GridSearchCV` (5-fold CV) |

**Random Forest hyperparameter grid searched:**
```
n_estimators:     [100, 200]
max_depth:        [None, 10, 20]
min_samples_split: [2, 5]
```

### 4. Evaluation

Models are compared using four metrics:
- **Accuracy** – Overall correctness
- **Precision** – Of predicted positives, how many are correct
- **Recall** – Of all actual positives, how many were caught
- **F1-score** – Harmonic mean of precision and recall (primary metric)

5-fold cross-validation was used to provide robust performance estimates for both models.

---

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~0.76 | ~0.76 | ~0.79 | ~0.77 |
| **Random Forest** | **~0.80** | **~0.80** | **~0.82** | **~0.81** |

> **Note:** Metrics are approximate and may vary slightly depending on the random seed and data split. **Random Forest** outperforms the baseline Logistic Regression, exceeding the F1-score ≥ 0.75 success criterion.

### Top Predictive Features (Random Forest)

1. `work_interfere` – Frequency that mental health interferes with work
2. `family_history` – Family history of mental illness
3. `benefits` – Employer-provided mental health benefits
4. `care_options` – Awareness of available care options
5. `anonymity` – Anonymity protection when seeking help

---

## Summary of Findings

- **Family history** of mental illness is among the strongest predictors of treatment-seeking behavior
- Employees with access to **mental health benefits** are significantly more likely to seek treatment
- **Workplace anonymity** and **supervisor support** are critical enablers of treatment-seeking
- **Work interference** — how often mental health issues impact daily work — is the top model feature
- **Random Forest** outperformed Logistic Regression, achieving an F1-score of ~0.81

### Deployment Potential

The trained Random Forest model can be deployed as an internal HR analytics tool to:
- Identify employee segments with lower treatment-seeking likelihood
- Trigger targeted wellness outreach programs
- Monitor changes in workplace mental health metrics over time

---

## Business Recommendations

| Finding | Recommendation |
|---------|----------------|
| Family history is a strong predictor | Increase mental health awareness for **all** employees, not just those with family history |
| Benefits availability increases treatment rates | Improve benefit communication and accessibility during onboarding and annually |
| Workplace anonymity matters | Protect confidentiality of all mental health disclosures |
| Supervisor support is critical | Train all managers in mental health first aid and communication |
| Work interference is highly predictive | Offer flexible work arrangements and Employee Assistance Programs (EAPs) |
| Remote workers may be underserved | Provide virtual mental health support and regular remote check-ins |

---

## Project Structure

```
tech-mental-health-classification/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── data/
│   └── survey.csv                     # OSMI Mental Health in Tech Survey data
├── notebooks/
│   └── mental_health_analysis.ipynb   # Full CRISP-DM analysis notebook
└── images/
    └── crisp_dm_diagram.png           # CRISP-DM framework diagram
```

📓 [View the Full Analysis Notebook](notebooks/mental_health_analysis.ipynb)

---

## Setup & Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebooks/mental_health_analysis.ipynb
```

**Requirements:** Python 3.8+, pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter

