"""
Drive Mode Agent - Manages vehicle driving dynamics and modes
Supports TechCar Model X and AutoDrive CX-7
"""

import logging
from strands import tool
from data.vehicle_systems import get_virtual_ecu

logger = logging.getLogger(__name__)


@tool
def drive_mode(
    action: str,
    mode: str = None,
    enable: bool = None
) -> str:
    """
    Control vehicle drive mode and dynamics settings for different driving conditions.
    Safety: Mode changes only allowed below 5 mph.

    Args:
        action: The drive mode action to perform. Options:
            - "set_mode": Set drive mode
            - "toggle_traction": Toggle traction control
            - "toggle_stability": Toggle stability control
            - "toggle_lane_assist": Toggle lane keeping assist
            - "toggle_cruise": Toggle adaptive cruise control
        mode: Drive mode for set_mode ("normal", "sport", "eco", "snow")
        enable: Enable/disable for toggle actions

    Returns:
        Confirmation of mode change with performance impact
    """
    ecu = get_virtual_ecu()
    current_mode = ecu.get_state("drive_mode")
    vehicle_info = ecu.get_state("vehicle_info")

    try:
        # Validate action parameter
        valid_actions = ["set_mode", "toggle_traction", "toggle_stability", "toggle_lane_assist", "toggle_cruise"]
        if action not in valid_actions:
            return f"Invalid action '{action}'. Valid actions: {', '.join(valid_actions)}"

        # Safety check for mode changes
        if action == "set_mode" and vehicle_info["speed"] > 5:
            return f"⚠️ Safety: Cannot change drive mode at {vehicle_info['speed']} mph. Please slow down below 5 mph."

        # Handle drive mode setting
        if action == "set_mode":
            if mode is None:
                return "Mode value required for set_mode action"
            valid_modes = ["normal", "sport", "eco", "snow"]
            if mode not in valid_modes:
                return f"Invalid mode '{mode}'. Valid modes: {', '.join(valid_modes)}"

            result = ecu.execute_command({
                "component": "drive_mode", "action": "set_mode", "value": mode
            })

            # Add mode-specific information
            mode_info = {
                "sport": """Sport mode engaged:
- Enhanced throttle response for quicker acceleration
- Firmer steering feel for better feedback
- Tighter suspension (AutoDrive CX-7)
- Engine sound enhancement active
- Fuel economy reduced by ~15%""",
                "eco": """Eco mode engaged:
- Optimized throttle for fuel efficiency
- Earlier upshifts for lower RPM
- Climate control optimization
- Regenerative braking enhanced
- Fuel economy improved by ~10-15%""",
                "snow": """Snow mode engaged:
- Gentle throttle response to prevent wheel spin
- Second gear starts for better traction
- Enhanced traction control
- Stability control maximized
- All-wheel drive optimized (if equipped)""",
                "normal": """Normal mode engaged:
- Balanced performance and comfort
- Standard throttle response
- Comfort-oriented suspension
- Optimal fuel economy
- All systems in default state"""
            }
            additional_info = mode_info.get(mode, "")
        # Handle traction control toggle
        elif action == "toggle_traction":
            if enable is None:
                enable = not current_mode["traction_control"]  # Toggle current state

            # Safety check for disabling while moving
            if not enable and vehicle_info["speed"] > 0:
                return "⚠️ Safety: Cannot disable traction control while vehicle is moving."

            result = ecu.execute_command({
                "component": "drive_mode", "action": "toggle_traction", "value": enable
            })
            additional_info = f"Traction control {'enabled' if enable else 'disabled (use caution!)'}"

        # Handle stability control toggle
        elif action == "toggle_stability":
            if enable is None:
                enable = not current_mode["stability_control"]  # Toggle current state

            # Safety check for disabling while moving
            if not enable and vehicle_info["speed"] > 0:
                return "⚠️ Safety: Cannot disable stability control while vehicle is moving."

            result = ecu.execute_command({
                "component": "drive_mode", "action": "toggle_stability", "value": enable
            })
            additional_info = f"Stability control {'enabled' if enable else 'disabled (expert drivers only!)'}"
        # Handle lane assist toggle
        elif action == "toggle_lane_assist":
            if enable is None:
                enable = not current_mode["lane_assist"]  # Toggle current state

            result = ecu.execute_command({
                "component": "drive_mode", "action": "toggle_lane_assist", "value": enable
            })
            additional_info = f"Lane keeping assist {'enabled' if enable else 'disabled'}"

        # Handle adaptive cruise toggle
        elif action == "toggle_cruise":
            if vehicle_info["speed"] < 25:
                return "Adaptive cruise control requires minimum speed of 25 mph"

            if enable is None:
                enable = not current_mode["adaptive_cruise"]  # Toggle current state

            result = ecu.execute_command({
                "component": "drive_mode", "action": "toggle_cruise", "value": enable
            })
            additional_info = f"Adaptive cruise control {'engaged' if enable else 'disengaged'}"

        else:
            return f"Unknown action '{action}'. Valid actions: {', '.join(valid_actions)}"

        # Process result
        if result["success"]:
            message = result["message"]

            # Add additional mode information if available
            if "additional_info" in locals():
                message = additional_info

            # Add vehicle context
            if vehicle_info["speed"] > 0:
                message += f"\n(Applied at {vehicle_info['speed']} mph)"

            # Log CAN signals
            if result.get("can_signals"):
                logger.debug(f"CAN signals: {result['can_signals']}")

            return message
        else:
            return f"Cannot change drive mode: {result['message']}"

    except Exception as e:
        logger.error(f"Drive mode error: {e}")
        return f"Sorry, I couldn't process that drive mode command. Current mode is {current_mode['current']}."
