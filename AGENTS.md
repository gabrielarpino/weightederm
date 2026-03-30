# AGENTS.md

## Project
Python package: weightederm

## Environment
- Use uv
- Install dependencies with:
  - uv sync

## Commands
- Run all tests:
  - uv run pytest -q
- Run one test file:
  - uv run pytest tests/test_smoke.py -q
- Lint:
  - uv run ruff check .
- Format check:
  - uv run ruff format --check .

## Workflow
- Use the `tdd` skill for implementation work
- Follow strict RED -> GREEN -> REFACTOR
- Keep diffs minimal
- Do not modify unrelated files
- Do not add dependencies unless necessary

## Structure
- Source: src/weightederm/
- Tests: tests/
