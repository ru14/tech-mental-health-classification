# Data Directory

## Source

| Field         | Value                                                                 |
|---------------|-----------------------------------------------------------------------|
| **Dataset**   | OSMI Mental Health in Tech Survey                                     |
| **Publisher** | Open Sourcing Mental Illness (OSMI)                                   |
| **URL**       | https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey     |
| **License**   | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)       |
| **Records**   | 1,259                                                                 |
| **Features**  | 27 columns                                                            |
| **Target**    | `treatment` — whether a respondent sought mental health treatment     |

## Files

| File          | Description                                          |
|---------------|------------------------------------------------------|
| `survey.csv`  | Raw survey responses as downloaded from Kaggle/OSMI  |

## Preprocessing Applied

All preprocessing is performed **inside the notebook**
(`notebooks/mental_health_analysis.ipynb`) — the CSV is never modified in
place. Key preprocessing steps:

1. **Age cleaning** — values outside 18–100 are treated as invalid and removed.
2. **Gender standardisation** — free-text responses mapped to
   `Male`, `Female`, `Other`.
3. **Missing value handling** — `self_employed` and `work_interfere` NaN values
   are imputed.
4. **Encoding** — categorical features are label-encoded or one-hot-encoded.
5. **Feature engineering** — `age_group` column derived from continuous `Age`.

## Privacy Note

The OSMI survey collects self-reported responses. Although the dataset is
publicly available and anonymised, please:

- Do **not** attempt to re-identify individuals.
- Do **not** add columns that could link responses to real people.
- Strip notebook outputs before committing
  (`nbstripout` pre-commit hook) to avoid accidentally publishing printed rows.

## Citation

> Open Sourcing Mental Illness (OSMI). *Mental Health in Tech Survey* (2014).
> Retrieved from https://osmihelp.org/research
