"""Live model monitoring: is the deployed model still operating on the data it was built for?

Offline test metrics say the model was good on held-out data. Monitoring asks the harder
production question: has the live input distribution drifted away from that reference, and is
the model still calibrated? :mod:`drift` provides population-stability and KS measures;
:class:`CalibrationTracker` bins predicted-vs-realized to keep an eye on calibration.
"""
from .drift import (CalibrationTracker, ks_statistic, population_stability_index,
                    drift_report, temporal_drift_report)

__all__ = ["CalibrationTracker", "ks_statistic", "population_stability_index",
           "drift_report", "temporal_drift_report"]
