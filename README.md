### Mental Health in Tech – Predicting Treatment-Seeking Behavior

**Raginee Upadhyaya**

#### Executive summary

This project applies the CRISP-DM framework and machine learning classification models to predict whether a tech industry employee will seek mental health treatment. Using the OSMI Mental Health in Tech Survey dataset (1,259 respondents), we trained and evaluated Logistic Regression, Random Forest, and Decision Tree classifiers. The analysis identifies key workplace and demographic factors — such as family history, access to mental health benefits, and supervisor support — that drive treatment-seeking behavior, and translates findings into actionable recommendations for tech organizations.

#### Rationale
Why should anyone care about this question?

Mental health issues are highly prevalent in the tech industry. Unaddressed mental health conditions lead to decreased productivity, increased absenteeism, and higher employee turnover. Identifying the factors that predict whether an employee will seek treatment enables organizations to design more effective benefit programs, reduce workplace stigma, and proactively support their workforce — ultimately benefiting both employees and business outcomes.

#### Research Question
What are you trying to answer?

Can we predict whether a tech employee will seek mental health treatment based on workplace and demographic factors? Which features (e.g., access to benefits, anonymity protections, supervisor support, family history) are most predictive of treatment-seeking behavior?

#### Data Sources
What data will you use to answer your question?

The dataset (`data/survey.csv`) is sourced from the [OSMI Mental Health in Tech Survey](https://osmihelp.org/research). It contains 1,259 responses from tech industry employees and includes the following key columns:

| Column | Description |
|--------|-------------|
| `Age` | Respondent age |
| `Gender` | Respondent gender |
| `Country` | Country of employment |
| `family_history` | Family history of mental illness (Yes/No) |
| `treatment` | **Target** – sought mental health treatment (Yes/No) |
| `work_interfere` | How often mental health interferes with work |
| `benefits` | Whether employer provides mental health benefits |
| `care_options` | Awareness of mental health care options |
| `anonymity` | Whether anonymity is protected when seeking help |
| `supervisor` | Comfort discussing mental health with supervisor |
| `no_employees` | Company size |
| `remote_work` | Whether the respondent works remotely |

#### Methodology
What methods are you using to answer the question?

This project follows the six phases of the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:

![CRISP-DM Diagram](images/crisp_dm_diagram.png)

1. **Business Understanding** – Define the prediction problem and success criteria
2. **Data Understanding** – Explore the dataset, check distributions and missing values
3. **Data Preparation** – Clean data, handle outliers, engineer features, encode categorical variables
4. **Modeling** – Train Logistic Regression (baseline), Random Forest, and Decision Tree with GridSearchCV hyperparameter tuning
5. **Evaluation** – Compare models using Accuracy, Precision, Recall, and F1-score
6. **Deployment & Recommendations** – Derive actionable business insights from model results

#### Results
What did your research find?

- **Family history** of mental illness is the strongest predictor of treatment-seeking behavior
- Employees with access to **mental health benefits** are significantly more likely to seek treatment
- **Workplace anonymity** and **supervisor support** are critical enablers of treatment-seeking
- **Logistic Regression**, **Random Forest**, and **Decision Tree** were compared; all models achieved competitive performance on Accuracy, Precision, Recall, and F1-score

#### Next steps
What suggestions do you have for next steps?

1. **Improve mental health benefit awareness** – Communicate available resources clearly during onboarding and annually
2. **Ensure confidentiality** – Guarantee anonymity for mental health disclosures to reduce stigma around seeking help
3. **Train supervisors** – Provide mental health first aid training for all managers
4. **Address stigma** – Normalize mental health conversations through company-wide culture initiatives
5. **Support remote workers** – Offer virtual mental health resources and regular check-ins for distributed teams
6. **Expand the dataset** – Incorporate more recent survey years and additional workplace variables to improve model generalizability

#### Outline of project

- [Link to Mental Health Analysis Notebook](notebooks/mental_health_analysis.ipynb)


##### Contact and Further Information

For questions or further information about this project, please reach out via the [GitHub repository](https://github.com/ru14/tech-mental-health-classification).

