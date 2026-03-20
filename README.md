# Mental Health in Tech – Treatment Prediction

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?logo=scikitlearn&logoColor=white)
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
├── README.md
├── requirements.txt
├── data/
│   └── survey.csv
├── images/
│   ├── crisp_dm_diagram.png
│   └── mental-health-in-tech.png
└── notebooks/
    └── mental_health_analysis.ipynb
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/ru14/tech-mental-health-classification.git
cd tech-mental-health-classification

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebooks/mental_health_analysis.ipynb
```

---

## Notebook

[**mental_health_analysis.ipynb**](notebooks/mental_health_analysis.ipynb) — Full CRISP-DM pipeline: data cleaning, EDA, modeling, evaluation, SHAP explainability, and business recommendations.

---

## Contact

For questions or feedback, reach out via the [GitHub repository](https://github.com/ru14/tech-mental-health-classification).

