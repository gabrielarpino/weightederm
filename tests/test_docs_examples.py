"""
Execute all tagged code blocks from docs/*.md.

A code block is included when its first non-blank line contains a comment
matching the pattern  # <TAG>_TEST: <name>  where TAG is one of:
  DOCS, ATTRS, CHOOSING

This mirrors the README_TEST_ convention used in test_readme_examples.py.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
_BLOCK_PATTERN = re.compile(r"```python\n(.*?)```", re.DOTALL)
_TAG_PATTERN = re.compile(r"#\s*(?:DOCS|ATTRS|CHOOSING)_TEST\s*:")


def _tagged_blocks_from_file(path: Path) -> list[tuple[str, str]]:
    """Return (tag_comment, source) pairs for every tagged block in *path*."""
    contents = path.read_text()
    blocks = []
    for match in _BLOCK_PATTERN.finditer(contents):
        block = match.group(1)
        first_line = next((ln for ln in block.splitlines() if ln.strip()), "")
        if _TAG_PATTERN.search(first_line):
            blocks.append((first_line.strip(), block))
    return blocks


def _all_tagged_blocks() -> list[tuple[str, str, str]]:
    """Return (filename, tag_comment, source) for every tagged block."""
    result = []
    for md_file in sorted(DOCS_DIR.glob("*.md")):
        for tag, source in _tagged_blocks_from_file(md_file):
            result.append((md_file.name, tag, source))
    return result


def test_docs_contain_expected_tagged_blocks():
    blocks = _all_tagged_blocks()
    tags = [tag for _, tag, _ in blocks]

    # user_guide.md
    assert any("minimal_workflow" in t for t in tags)
    assert any("fixed_estimator" in t for t in tags)
    assert any("cv_estimator" in t for t in tags)
    assert any("penalty_usage" in t for t in tags)
    assert any("predict_usage" in t for t in tags)
    assert any("sklearn_pipeline" in t for t in tags)

    # fitted_attributes.md
    assert any("changepoints" in t for t in tags)
    assert any("cv_results" in t for t in tags)

    # choosing_estimator.md
    assert any("fixed_known" in t for t in tags)
    assert any("cv_unknown" in t for t in tags)


def test_docs_tagged_blocks_execute_successfully():
    blocks = _all_tagged_blocks()
    assert len(blocks) > 0, "No tagged blocks found in docs/"

    for filename, tag, source in blocks:
        namespace: dict = {"__name__": "__main__"}
        try:
            exec(source, namespace, namespace)  # noqa: S102
        except Exception as exc:
            raise AssertionError(
                f"Docs code block failed.\n"
                f"  File   : {filename}\n"
                f"  Tag    : {tag}\n"
                f"  Error  : {exc}"
            ) from exc
