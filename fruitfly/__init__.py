"""FruitFlyLuau -- connectome-grounded spiking simulation that we train with
reinforcement and plasticity, aiming (much later) at learning Luau.

Honesty contract, restated in code: ``fruitfly`` simulates a *structural* wiring
diagram (FlyWire FAFB v783) with a *computational* neuron model, and "learning"
here means one thing only -- changes to the synaptic weight array made by an
explicit plasticity rule. See LIMITATIONS.md.
"""

from .version import __version__

__all__ = ["__version__"]
