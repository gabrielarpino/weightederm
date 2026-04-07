from __future__ import annotations

import re
from pathlib import Path


def _readme_test_blocks() -> list[str]:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    contents = readme.read_text()
    blocks = re.findall(r"```python\n(.*?)```", contents, flags=re.DOTALL)
    return [block for block in blocks if "README_TEST_" in block]


def test_readme_contains_three_tested_example_blocks():
    blocks = _readme_test_blocks()

    assert len(blocks) == 3


def test_readme_references_existing_header_image():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    contents = readme.read_text()

    assert "![weightederm header](assets/raw_marginal_plot_5.svg)" in contents
    assert (readme.parent / "assets" / "raw_marginal_plot_5.svg").exists()


def test_readme_example_blocks_execute_successfully():
    for block in _readme_test_blocks():
        namespace = {"__name__": "__main__"}
        exec(block, namespace, namespace)
