# Changelog

## Unreleased

### Tooling
- CI runs `ruff check`, `ruff format --check` and `ty check` (pinned to 0.0.107) in a separate `lint` job, which also gates the release build; `pre-commit` runs ruff locally and rejects AI-attribution trailers in commit messages.
- ruff enforces complexity (`C901`, `max-complexity = 15`), docstrings and the same rule set as `extended-lengyel`, over `radas/` and `tests/`.
- `pytest` fails below 81% coverage (`--cov-fail-under` in `pyproject.toml`).
- Dependabot opens weekly updates for pip/poetry and GitHub Actions.
- Pull requests are checked by `actions/dependency-review-action` (fails on moderate or higher severity advisories in added or bumped dependencies).
