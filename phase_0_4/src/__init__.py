"""
Phase 0.4: Proxy Approximation Validity Gate
Experimental infrastructure for validating TCR's core theoretical assumption.
"""

__version__ = "1.0.0"
__author__ = "TCR Development Team"

from .utility_computer import UtilityComputer, AggregateUtilityScore
from .gain_measurer import StratifiedGainSampler, GainAnalyzer
from .correlation_validator import CorrelationValidator, CorrelationReporter
from .visualizer import CorrelationVisualizer
from .main import Phase04Orchestrator

__all__ = [
    'UtilityComputer',
    'AggregateUtilityScore',
    'StratifiedGainSampler',
    'GainAnalyzer',
    'CorrelationValidator',
    'CorrelationReporter',
    'CorrelationVisualizer',
    'Phase04Orchestrator',
]
