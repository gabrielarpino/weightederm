---
name: tdd
description: Strict test-driven development workflow using RED->GREEN->REFACTOR cycles. Use for all implementation work.
---

# TDD Discipline

## Core Principle
Write tests first. Implement minimal code. Refactor safely.

## Pre-Implementation Checklist
Before writing any production code:
1. Write the test
2. Run it
3. Confirm it fails for the expected reason

If not, fix the test first.

## RED -> GREEN -> REFACTOR

### RED
Write tests covering:
- Happy paths
- Edge cases
- Errors

Run the smallest relevant test selection.
The newly added or targeted test must fail before implementation.

### GREEN
Implement the smallest atomic change to pass the failing test.

Rules:
- One behavior at a time
- If unrelated tests pass, split the change smaller

After each step:
1. Run targeted tests
2. If green, run full suite

### REFACTOR
Only refactor when tests are green.
Run tests after every refactor.

## Test Rules
- Never weaken tests to make code pass
- Never delete tests to make code pass

If a test seems wrong:
1. Revert implementation
2. Rewrite test
3. Confirm it fails
4. Re-implement

## Contract Changes
If intentionally changing API:
1. Update existing tests first
2. Ensure they fail
3. Then implement

## Code Quality
- No `Any` or `type: ignore` unless necessary
- No bare `assert` in production code
- Prefer explicit logic
- Fix root causes, not symptoms

## Final Verification
Before finishing:
- Run full test suite
- Run lint

Report:
- tests added
- failing state
- implementation change
- commands run
- final status
