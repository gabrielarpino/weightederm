"""Public estimator interface for the weightederm package."""

from weightederm._cv_estimators import WERMHuberCV, WERMLeastSquaresCV, WERMLogisticCV
from weightederm._huber import WERMHuber
from weightederm._least_squares import WERMLeastSquares
from weightederm._logistic import WERMLogistic

__all__ = [
    "WERMHuber",
    "WERMHuberCV",
    "WERMLeastSquares",
    "WERMLeastSquaresCV",
    "WERMLogistic",
    "WERMLogisticCV",
    "__version__",
]
__version__ = "0.1.0"
