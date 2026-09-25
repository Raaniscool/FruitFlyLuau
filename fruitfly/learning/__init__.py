"""Reinforcement machinery: reward ledger + dopamine-like modulator.

Plasticity rules live in :mod:`fruitfly.neuro.plasticity` because they mutate the
neuron model's weight array; they are re-exported here for convenience.
"""

from ..neuro.plasticity import (
    FrozenWeights,
    HebbianTrace,
    PlasticityContext,
    PlasticityRule,
    RewardModulatedSTDP,
    STDP,
    SynapseState,
    available_rules,
    build_rule,
    register_rule,
)
from .modulator import DopamineLikeModulator
from .reward import RewardEvent, RewardSystem

__all__ = [
    "RewardSystem", "RewardEvent", "DopamineLikeModulator",
    "PlasticityContext", "PlasticityRule", "SynapseState",
    "STDP", "RewardModulatedSTDP", "HebbianTrace", "FrozenWeights",
    "build_rule", "available_rules", "register_rule",
]
