"""Experiment framework: environments, runner, metrics, checkpoints."""

from .base import Environment, Trial, register_env, build_environment, available_environments
from .runner import ExperimentRunner, TrainOutcome, TrialResult
from .checkpoint import save_checkpoint, load_checkpoint, restore_connectome
from .metrics import MetricsLogger, RunningStats, chance_level, binom_p_above_chance
from . import tasks  # registers exp001..exp005

__all__ = ["Environment", "Trial", "register_env", "build_environment", "available_environments",
           "ExperimentRunner", "TrainOutcome", "TrialResult", "save_checkpoint", "load_checkpoint",
           "restore_connectome", "MetricsLogger", "RunningStats", "chance_level", "binom_p_above_chance"]
