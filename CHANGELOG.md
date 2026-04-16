# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Repository security hardening: `.gitignore`, `SECURITY.md`, `LICENSE`,
  Dependabot, pinned GitHub Actions SHAs, `pip-audit` CI scan.
- Contribution infrastructure: `CONTRIBUTING.md`, `CODEOWNERS`, PR template,
  issue templates.
- Developer tooling: `pre-commit` config (`nbstripout`, `detect-secrets`,
  `trailing-whitespace`), `pyproject.toml`, `requirements-dev.txt`.
- Data provenance documentation (`data/README.md`).
- `.env.example` template for future environment variables.
- `.gitattributes` for consistent line endings and CSV diff rendering.
- `VERSION` file for reproducibility tracking.
- Lint & format GitHub Actions workflow (`lint.yml`).

---

## [1.0.0] — 2026-04-01

### Added
- Initial CRISP-DM pipeline notebook (`notebooks/mental_health_analysis.ipynb`).
- Logistic Regression, Random Forest, and Decision Tree classifiers.
- SHAP explainability (summary plot + force plot).
- Notebook validation CI workflow (`validate_notebook.yml`).
- Project README with methodology, results, and getting-started instructions.

[Unreleased]: https://github.com/ru14/tech-mental-health-classification/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ru14/tech-mental-health-classification/releases/tag/v1.0.0
