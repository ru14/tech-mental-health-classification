# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| main    | :white_check_mark: |

## Reporting a Vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

If you discover a security issue in this repository — including problems with
dependencies, data handling, or model artefacts — please report it privately:

1. Go to the repository's **Security** tab on GitHub.
2. Click **"Report a vulnerability"** to open a private advisory.
3. Provide as much detail as possible:
   - A description of the vulnerability and its potential impact.
   - Steps to reproduce or a proof-of-concept.
   - Any suggested mitigations.

We will acknowledge your report within **48 hours** and aim to resolve
confirmed vulnerabilities within **7 days**.

## Scope

Although this is a data-science / research project, the following are in
scope for security reports:

- Secrets or credentials accidentally committed to the repository.
- Personally Identifiable Information (PII) leaked through notebook outputs.
- Malicious or compromised dependencies listed in `requirements.txt`.
- Supply-chain issues in GitHub Actions workflows.
- Insecure handling of survey data.

## Out of Scope

- Issues in upstream packages that are already publicly disclosed and tracked
  by Dependabot.
- Theoretical vulnerabilities with no practical exploit path.

Thank you for helping keep this project safe.
