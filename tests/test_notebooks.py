import json
from pathlib import Path


def test_reference_like_benchmark_notebook_exists_and_mentions_m1_m2_m3() -> None:
    notebook_path = Path("notebooks/reference_like_m123_benchmarks.ipynb")

    assert notebook_path.exists()

    notebook = json.loads(notebook_path.read_text())
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    combined = "\n".join(sources)

    assert "M1" in combined
    assert "M2" in combined
    assert "M3" in combined
    assert "run_benchmark" in combined
    assert "run_benchmark_unknown" in combined
    assert "known number of change points" in combined
    assert "unknown number of change points" in combined
    assert "MCSCAN_MODE" in combined
    assert "inferchange" in combined
    assert "INSTALL_MCSCAN_DEPS" in combined
    assert "LEAST_SQUARES_FIT_SOLVER" in combined
    assert "predicted_num_chgpts" in combined
    assert "mean_predicted_num_chgpts" in combined
    assert "num_infinite_hausdorff" in combined
    assert "num_nan_hausdorff" in combined
    assert "M1_NOTEBOOK_DELTA_RATIOS" in combined
    assert "7.0" in combined
    assert "use_base_loss_for_cv" in combined
