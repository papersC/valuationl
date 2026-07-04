"""Real-transaction accuracy experiment for the valuation pipeline.

Loads real residential sales (NYC DOF annualized sales; the loader is
source-agnostic and a Dubai DLD loader can be dropped in behind the same
DataFrame contract), fits a gradient-boosted hedonic adjuster behind the
sales-comparison interface, and measures point accuracy, uniformity, and
split-conformal interval calibration under temporal (walk-forward) evaluation.
"""
