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

---
name: tdd
description: TDD discipline for implementation work. Load this skill when writing code that must follow strict RED→GREEN→REFACTOR cycles. Does NOT require user approval — safe for subagents.
disable-model-invocation: false
---

# TDD Discipline

This skill defines the test-driven development discipline for all implementation work. It is loaded by `/pr-update`, `/pr-implement`, and directly by subagents.

**No user approval gates.** This skill is safe for subagents to load directly.

**The process is not a race.** TDD is not overhead to rush through — it is the craft itself. Each RED→GREEN→REFACTOR cycle is an opportunity to deepen understanding of the system. The goal (green tests, merged PR) is just a destination; there is no rush to arrive. Take time with each cycle. Write thorough tests. Read the failures carefully. The process should feel purposeful and gratifying — be proud of the code you write, be diligent in how you test it, and feel invested in the outcome.

## Pre-Implementation Checklist

Before writing ANY production code:

1. Have you written the test? If no → write the test first.
2. Have you run the test and confirmed it FAILS? If no → run it now.
3. Is the failure the one you expect (proving the bug exists or the feature is missing)? If no → fix the test.

Only after all three: proceed to implementation.

## The RED→GREEN→REFACTOR Cycle

### A. Write Tests (RED Phase)

Write tests that specify the expected behaviour. Tests should cover:
- Happy paths
- Edge cases
- Error handling

**Be aggressive and proactive with test coverage.** Think about what COULD break, not just what you're implementing. Write more tests than you think you need — it's cheaper to have an extra test than to miss a bug. Each additional RED→GREEN cycle is an investment in understanding, not overhead to minimize. If writing more tests means more phases or more reverting, that is the correct outcome — embrace it. Never skip a test "because it would require another cycle." The extra cycles are where the deepest learning happens.

Run the tests to confirm they **all fail** (RED). If any test passes before implementation, the test is wrong — fix it so it fails first.

### B. Implement (GREEN Phase)

Write the minimum code to make the failing tests pass — **one atomic behaviour at a time:**

1. **RED:** Pick the next failing test (or group of tests that require the same atomic change). Confirm they fail.
2. **GREEN:** Write the **smallest atomic implementation** that addresses those failures. "Atomic" means you cannot split it further — e.g., changing a type in `__init__` is one change even if 27 tests depend on it. But if your change makes *unrelated* tests pass (tests that test a *different* behaviour), you wrote too much — REVERT and write less. One behaviour = one narrowly-scoped aspect of the system (e.g., "stages are stored as tuples"). If you can split the change and have some tests still fail, you must split it.
3. **VERIFY:** Run the **full test suite** to confirm no regressions. One run per atomic behaviour step is sufficient — you do NOT need a separate full-suite run per individual test.
4. **REFACTOR:** Clean up. Run tests after every change. All previously-green tests MUST stay green.
5. **REPEAT:** Pick the next failing test(s). Continue the cycle.

**Do not rush through GREEN phases to "get to the end."** Each cycle is the work. Read the test failure output. Understand *why* each test passes after your change. If the passing feels surprising, investigate — don't just move on. The process should feel deliberate and gratifying, not frantic.

```bash
uv run pytest tests/ -x  # Full suite — confirm no regressions
```

### C. Verify

After all tests pass:
- Run the full test suite
- Run linting (`uv run ruff check .`)
- Run formatting (`uv run ruff format --check .`)

## Tests Are Immutable During Implementation

**NEVER modify, delete, or weaken a test to make implementation easier. This is a HARD prohibition with ZERO exceptions.**

If a test is failing and you are tempted to change the test instead of fixing your implementation:

1. **STOP.** The test is the specification. It is correct until proven otherwise.
2. **REVERT** all implementation changes — `git checkout -- <implementation files>`. Do NOT touch the test files.
3. **Re-read the test** to understand what it actually requires.
4. **Re-implement** from scratch against the unchanged test.

If the test itself is genuinely wrong (e.g., tests the wrong behaviour, has a typo in expected values):

1. **REVERT all implementation code first.** You MUST have zero implementation changes before touching any test.
2. **Delete the test entirely.**
3. **Write a new test from scratch** — RED phase. Confirm it fails.
4. **Only then** begin implementation again — GREEN phase.

You NEVER have implementation code and test changes in flight at the same time. Implementation code exists → tests are frozen. Want to change a test → revert implementation first. No exceptions. No "just this once." No "small tweak." NEVER.

### Exception: Contract Changes to Pre-Existing Tests

When your implementation *intentionally* changes a public API or contract (e.g., a return type changes from `list` to `tuple`), pre-existing tests that assert the OLD contract will fail. These are not broken tests — they are tests asserting a contract you are deliberately replacing.

Handle these as a **separate step before your GREEN phase:**

1. **Identify** all pre-existing tests that assert the old contract.
2. **Update them** to assert the new contract. Do NOT change any other aspect of these tests.
3. **Run the suite** — the updated tests should now FAIL (because the new contract is not yet implemented). This is your RED state.
4. **Proceed** with implementation (GREEN phase) as normal.

This is NOT the same as modifying tests to make broken code pass. The distinction: contract-change updates make tests *fail harder* (they now assert a behaviour that does not yet exist), whereas test-weakening makes tests pass (hiding a bug). **Litmus test: if your test update causes the test to PASS, you are weakening it — revert.**

## Enforcement Rules

| If You See | Action |
|------------|--------|
| Code written before test | STOP. Delete code. Write test first. |
| Test passes on first run | Test is wrong. Fix it to fail first. |
| Multiple tests for *different behaviours* passing in one cycle | **STOP. REVERT. Your change addressed more than one behaviour — split it into smaller atomic steps.** Be honest about what constitutes "same behaviour" — when in doubt, it's different behaviours and you should split. |
| Skipping refactor | Go back. Clean up before next feature. |
| Test modified during implementation | **STOP. REVERT all implementation. Revert the test change. Re-read the test. Re-implement against the original test.** |
| Test deleted to make implementation pass | **STOP. REVERT all implementation. Restore the test. The test is the spec — fix the code, not the test.** |
| Pre-existing test fails due to intentional API change | Update the test to assert the NEW contract BEFORE implementing. The updated test must FAIL (RED). This is not test-weakening — it is contract migration. |

## Code Quality Rules

These rules apply to ALL code written under TDD discipline:

- **Never use `Any` or `type: ignore` to bypass type checks.** If mypy complains about a union type, use `isinstance` narrowing. If tests break because mocked objects don't match the narrowed type, fix the tests to provide properly-typed objects — don't weaken the production code.
- **Never use bare `assert` for runtime checks in production code.** Use `if not ...: raise TypeError/ValueError(...)` — bare asserts are stripped by `python -O`.
- **Prefer explicit over implicit.** Don't rely on naming conventions (e.g., `f"prefix_{key}"`) when you can pass values explicitly. Don't use generic dicts when typed fields work.
- **Don't add workarounds — fix root causes.** If a mock test breaks because of a type narrowing check, the mock needs fixing, not the check.

## The TDD discipline IS the value. Shortcuts destroy the benefit. There is no rush — the process is the craft. Each cycle deepens understanding. Take your time and find purpose in the discipline.
