"""Real-transaction accuracy experiment for the valuation pipeline.

Loads real residential resales (the reported study uses the Dubai Land
Department open transaction feed via ``loader_dubai``; the loader is
source-agnostic and the NYC DOF loader remains available behind the same
DataFrame contract), fits a gradient-boosted hedonic adjuster behind the
sales-comparison interface, and measures point accuracy, uniformity, and
split-conformal interval calibration under temporal (walk-forward) evaluation.
"""
