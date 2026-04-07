from weightederm._cv_estimators import WERMHuberCV, WERMLeastSquaresCV, WERMLogisticCV


def test_consolidated_cv_estimators_module_exports_all_three_cv_estimators():
    assert WERMLeastSquaresCV.__name__ == "WERMLeastSquaresCV"
    assert WERMHuberCV.__name__ == "WERMHuberCV"
    assert WERMLogisticCV.__name__ == "WERMLogisticCV"
