# Mental Health in Tech – Treatment Prediction

[![Validate Notebook](https://github.com/ru14/tech-mental-health-classification/actions/workflows/validate_notebook.yml/badge.svg)](https://github.com/ru14/tech-mental-health-classification/actions/workflows/validate_notebook.yml)
[![Security Scan](https://github.com/ru14/tech-mental-health-classification/actions/workflows/security_scan.yml/badge.svg)](https://github.com/ru14/tech-mental-health-classification/actions/workflows/security_scan.yml)
[![Lint](https://github.com/ru14/tech-mental-health-classification/actions/workflows/lint.yml/badge.svg)](https://github.com/ru14/tech-mental-health-classification/actions/workflows/lint.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Can we predict whether a tech employee will seek mental health treatment based on workplace and demographic factors?**

**Author:** Raginee Upadhyaya

---

## Executive Summary

This project applies the **CRISP-DM** framework and machine learning to predict treatment-seeking behavior among tech employees. Using the [OSMI Mental Health in Tech Survey](https://osmihelp.org/research) (1,259 respondents), we trained Logistic Regression, Random Forest, and Decision Tree classifiers, then used **SHAP** to explain which workplace and demographic factors drive predictions. The analysis delivers actionable recommendations for tech organizations seeking to improve mental health support.

---

## Dataset

**Source:** [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey) &nbsp;|&nbsp; **Records:** 1,259 &nbsp;|&nbsp; **Features:** 27

| Column | Description |
|--------|-------------|
| `Age` | Respondent age |
| `Gender` | Respondent gender |
| `family_history` | Family history of mental illness (Yes/No) |
| `treatment` | **Target** – sought mental health treatment (Yes/No) |
| `work_interfere` | How often mental health interferes with work |
| `benefits` | Whether employer provides mental health benefits |
| `care_options` | Awareness of mental health care options |
| `anonymity` | Whether anonymity is protected when seeking help |
| `supervisor` | Comfort discussing mental health with supervisor |
| `no_employees` | Company size |
| `remote_work` | Whether the respondent works remotely |

---

## Methodology

This project follows the six phases of **CRISP-DM**:

![CRISP-DM Diagram](images/crisp_dm_diagram.png)

| Phase | Description |
|-------|-------------|
| **1. Business Understanding** | Define prediction problem and success criteria (F1 ≥ 0.75) |
| **2. Data Understanding** | Explore distributions, missing values, and class balance |
| **3. Data Preparation** | Clean ages, standardize gender, handle missing values, encode features, engineer age groups |
| **4. Modeling** | Logistic Regression (baseline), Random Forest, and Decision Tree with GridSearchCV tuning |
| **5. Evaluation** | Compare models on Accuracy, Precision, Recall, and F1-score; 5-fold cross-validation |
| **6. Deployment** | SHAP explainability + actionable business recommendations |

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~0.80 | ~0.80 | ~0.82 | ~0.81 |
| **Random Forest** | **~0.80** | **~0.79** | **~0.84** | **~0.81** |
| Decision Tree | ~0.75 | ~0.76 | ~0.78 | ~0.77 |

- All models exceeded the F1 ≥ 0.75 success criterion
- Random Forest selected as the primary model for its strong recall and ensemble stability

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) provides transparent, model-agnostic feature attributions:

- **Global importance** — `shap.summary_plot` reveals which features matter most across all predictions
- **Local explanations** — `shap.force_plot` shows exactly why the model predicted treatment (or not) for an individual employee

**Top predictive features (SHAP-ranked):**

| Rank | Feature | Direction |
|------|---------|-----------|
| 1 | `family_history` | Employees with family history are far more likely to seek treatment |
| 2 | `work_interfere` | Higher work interference → higher treatment likelihood |
| 3 | `benefits` | Access to benefits increases treatment-seeking |
| 4 | `care_options` | Awareness of care options is a strong positive driver |
| 5 | `anonymity` | Protected anonymity encourages treatment-seeking |

---

## Actionable Recommendations

| Finding | Recommendation |
|---------|----------------|
| Family history is the strongest predictor | Increase awareness for **all** employees, not just those with family history |
| Benefits availability increases treatment rates | Communicate mental health benefits clearly during onboarding and annually |
| Workplace anonymity matters | Guarantee confidentiality of mental health disclosures |
| Supervisor support is critical | Provide mental health first-aid training for all managers |
| Work interference is highly predictive | Offer flexible work arrangements and Employee Assistance Programs |
| Remote workers may lack access | Provide virtual mental health support and regular check-ins |

---

## Project Structure

```
tech-mental-health-classification/
├── .env.example                  # Environment variable template
├── .gitattributes                # Line-ending & diff settings
├── .gitignore
├── .pre-commit-config.yaml       # Pre-commit hooks (nbstripout, detect-secrets, black)
├── .python-version               # Pinned Python version (3.11)
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── SECURITY.md
├── VERSION
├── pyproject.toml                # Tool config (black, isort, pytest)
├── requirements.txt              # Pinned production dependencies
├── requirements-dev.txt          # Dev/CI dependencies
├── .github/
│   ├── CODEOWNERS
│   ├── dependabot.yml
│   ├── pull_request_template.md
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── security.md
│   └── workflows/
│       ├── validate_notebook.yml # Notebook execution CI
│       ├── lint.yml              # Format & quality checks
│       └── security_scan.yml    # pip-audit CVE scan
├── data/
│   ├── README.md                 # Data provenance & privacy notes
│   └── survey.csv
├── images/
│   └── *.png
└── notebooks/
    └── mental_health_analysis.ipynb
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/ru14/tech-mental-health-classification.git
cd tech-mental-health-classification

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install production dependencies (pinned)
pip install -r requirements.txt

# (Optional) Install dev tools and pre-commit hooks
pip install -r requirements-dev.txt
pre-commit install

# Launch the notebook
jupyter notebook notebooks/mental_health_analysis.ipynb
```

---

## Notebook

[**mental_health_analysis.ipynb**](notebooks/mental_health_analysis.ipynb) — Full CRISP-DM pipeline: data cleaning, EDA, modeling, evaluation, SHAP explainability, and business recommendations.

---

## Contact

For questions or feedback, reach out via the [GitHub repository](https://github.com/ru14/tech-mental-health-classification).

---

## Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md)
for branch naming conventions, commit message format, and the PR process.

## Security

To report a vulnerability, follow the instructions in
[`SECURITY.md`](SECURITY.md). Please **do not** open a public issue for
security matters.

