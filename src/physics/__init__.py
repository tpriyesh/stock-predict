"""
Physics-Inspired Models for Stock Prediction

This module implements physics concepts adapted for financial markets:
- Momentum Conservation: Objects in motion stay in motion
- Spring Reversion: Hooke's Law for mean reversion
- Energy Clustering: Thermodynamics for volatility
- Network Propagation: Wave mechanics for sector correlations
"""

from .momentum_conservation import MomentumConservationModel, MomentumResult
from .spring_reversion import SpringReversionModel, SpringResult
from .energy_clustering import EnergyClusteringModel, EnergyResult
from .network_propagation import NetworkPropagationModel, NetworkResult
from .physics_engine import PhysicsEngine, PhysicsScore

__all__ = [
    'MomentumConservationModel',
    'MomentumResult',
    'SpringReversionModel',
    'SpringResult',
    'EnergyClusteringModel',
    'EnergyResult',
    'NetworkPropagationModel',
    'NetworkResult',
    'PhysicsEngine',
    'PhysicsScore',
]
