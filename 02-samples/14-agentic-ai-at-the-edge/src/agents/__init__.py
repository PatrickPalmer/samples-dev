"""
Edge AI Agents - Vehicle Cockpit Control System

Structure:
- cockpit/: Direct vehicle control (climate, windows, seats, lights, drive modes)
- tools/: Shared tools and utilities (voice input, image analysis, model selection)
"""

# Cockpit Control Agents
from .cockpit import climate_control, window_control, seat_control, lighting_control, drive_mode

# Tools
from .tools import select_model

__all__ = [
    # Cockpit Control
    "climate_control",
    "window_control",
    "seat_control",
    "lighting_control",
    "drive_mode",
    # Tools
    "select_model",
]
