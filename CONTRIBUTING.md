# Contributing Guide

Thank you for taking the time to contribute! Please read this guide before
opening an issue or submitting a pull request.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Branch Naming](#branch-naming)
4. [Commit Messages](#commit-messages)
5. [Pull Request Process](#pull-request-process)
6. [Development Setup](#development-setup)
7. [Running the Notebook](#running-the-notebook)
8. [CI Checks](#ci-checks)
9. [Reporting Security Issues](#reporting-security-issues)

---

## Code of Conduct

By participating in this project you agree to be respectful and constructive.
Harassment or discrimination of any kind will not be tolerated.

---

## Getting Started

1. **Fork** the repository and clone your fork.
2. Create a virtual environment and install dependencies (see
   [Development Setup](#development-setup)).
3. Create a new branch from `main` following the
   [branch naming](#branch-naming) convention.
4. Make your changes, then open a pull request against `main`.

---

## Branch Naming

Use the following prefixes:

| Prefix      | Use for                                      |
|-------------|----------------------------------------------|
| `feat/`     | New features or enhancements                 |
| `fix/`      | Bug fixes                                    |
| `chore/`    | Tooling, CI, dependency updates              |
| `docs/`     | Documentation-only changes                   |
| `refactor/` | Code restructuring without behaviour changes |
| `security/` | Security-related changes                     |

**Examples:**

```
feat/add-xgboost-model
fix/age-cleaning-edge-cases
docs/update-readme-badges
chore/pin-dependency-versions
```

---

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/)
specification:

```
<type>(<optional scope>): <short imperative description>

[optional body]

[optional footer — e.g. Closes #42]
```

**Examples:**

```
feat(notebook): add XGBoost classifier with GridSearchCV tuning
fix(data): handle NaN ages outside 18–100 range
chore(deps): bump scikit-learn from 1.4.2 to 1.5.0
docs(readme): add venv setup instructions
```

---

## Pull Request Process

1. Ensure your branch is up-to-date with `main`.
2. Strip notebook outputs before pushing:
   ```bash
   nbstripout notebooks/mental_health_analysis.ipynb
   ```
   (The `pre-commit` hook does this automatically if installed.)
3. Fill in the **PR template** completely — incomplete PRs may be closed.
4. At least **one approving review** is required before merging.
5. All CI checks must pass before merge.
6. Update `CHANGELOG.md` for user-facing changes.

---

## Development Setup

```bash
# 1. Clone your fork
git clone https://github.com/<your-username>/tech-mental-health-classification.git
cd tech-mental-health-classification

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install production dependencies
pip install -r requirements.txt

# 4. Install development dependencies
pip install -r requirements-dev.txt

# 5. Install pre-commit hooks
pre-commit install

# 6. (Optional) Copy the environment template
cp .env.example .env
```

---

## Running the Notebook

```bash
jupyter notebook notebooks/mental_health_analysis.ipynb
```

To execute the full notebook non-interactively (mirrors CI):

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=600 \
  notebooks/mental_health_analysis.ipynb \
  --output /tmp/executed_notebook.ipynb
```

---

## CI Checks

| Workflow              | Trigger          | What it does                            |
|-----------------------|------------------|-----------------------------------------|
| `validate_notebook`   | push / PR        | Executes the notebook end-to-end        |
| `lint`                | push / PR        | Runs `nbstripout --verify`, formatting  |
| `security_scan`       | push / PR / cron | `pip-audit` CVE scan on dependencies    |

All three must pass before a PR can be merged.

---

## Reporting Security Issues

Please **do not** open a public GitHub issue for security vulnerabilities.
Follow the instructions in [`SECURITY.md`](SECURITY.md).
