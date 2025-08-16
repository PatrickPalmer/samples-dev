"""
Drive Mode Agent - Manages vehicle driving dynamics and modes
Supports TechCar Model X and AutoDrive CX-7
"""

import logging
from strands import tool
from ...data.vehicle_systems import get_virtual_ecu

logger = logging.getLogger(__name__)


@tool
def drive_mode(command: str) -> str:
    """
    Control vehicle drive mode and dynamics settings for different driving conditions.
    Safety: Mode changes only allowed below 5 mph.

    Examples:
    - "Switch to sport mode"
    - "Enable eco mode"
    - "Set to snow mode"
    - "Turn on traction control"
    - "I want better fuel economy" (will enable eco mode)
    - "It's snowing" (will enable snow mode)

    Args:
        command: Natural language drive mode request

    Returns:
        Confirmation of mode change with performance impact
    """
    ecu = get_virtual_ecu()
    current_mode = ecu.get_state("drive_mode")
    vehicle_info = ecu.get_state("vehicle_info")

    # Safety check for mode changes
    if vehicle_info["speed"] > 5 and "mode" in command.lower():
        return f"⚠️ Safety: Cannot change drive mode at {vehicle_info['speed']} mph. Please slow down below 5 mph."

    command_lower = command.lower()

    try:
        # Drive mode selection
        if any(word in command_lower for word in ["sport", "sporty", "performance", "fun"]):
            result = ecu.execute_command(
                {"component": "drive_mode", "action": "set_mode", "value": "sport"}
            )
            additional_info = """
Sport mode engaged:
- Enhanced throttle response for quicker acceleration
- Firmer steering feel for better feedback
- Tighter suspension (AutoDrive CX-7)
- Engine sound enhancement active
- Fuel economy reduced by ~15%"""

        elif any(
            word in command_lower
            for word in ["eco", "economy", "efficient", "save fuel", "better fuel"]
        ):
            result = ecu.execute_command(
                {"component": "drive_mode", "action": "set_mode", "value": "eco"}
            )
            additional_info = """
Eco mode engaged:
- Optimized throttle for fuel efficiency
- Earlier upshifts for lower RPM
- Climate control optimization
- Regenerative braking enhanced
- Fuel economy improved by ~10-15%"""

        elif any(
            word in command_lower
            for word in ["snow", "snowing", "ice", "icy", "slippery", "winter"]
        ):
            result = ecu.execute_command(
                {"component": "drive_mode", "action": "set_mode", "value": "snow"}
            )
            additional_info = """
Snow mode engaged:
- Gentle throttle response to prevent wheel spin
- Second gear starts for better traction
- Enhanced traction control
- Stability control maximized
- All-wheel drive optimized (if equipped)"""

        elif any(word in command_lower for word in ["normal", "comfort", "regular", "standard"]):
            result = ecu.execute_command(
                {"component": "drive_mode", "action": "set_mode", "value": "normal"}
            )
            additional_info = """
Normal mode engaged:
- Balanced performance and comfort
- Standard throttle response
- Comfort-oriented suspension
- Optimal fuel economy
- All systems in default state"""

        # Traction control
        elif "traction" in command_lower:
            if "off" in command_lower or "disable" in command_lower:
                if vehicle_info["speed"] > 0:
                    return "⚠️ Safety: Cannot disable traction control while vehicle is moving."
                value = False
            elif "on" in command_lower or "enable" in command_lower:
                value = True
            else:
                value = not current_mode["traction_control"]  # Toggle

            result = ecu.execute_command(
                {"component": "drive_mode", "action": "toggle_traction", "value": value}
            )
            additional_info = (
                f"Traction control {'enabled' if value else 'disabled (use caution!)'}"
            )

        # Stability control
        elif "stability" in command_lower or "esc" in command_lower:
            if "off" in command_lower or "disable" in command_lower:
                if vehicle_info["speed"] > 0:
                    return "⚠️ Safety: Cannot disable stability control while vehicle is moving."
                value = False
            elif "on" in command_lower or "enable" in command_lower:
                value = True
            else:
                value = not current_mode["stability_control"]  # Toggle

            result = ecu.execute_command(
                {"component": "drive_mode", "action": "toggle_stability", "value": value}
            )
            additional_info = (
                f"Stability control {'enabled' if value else 'disabled (expert drivers only!)'}"
            )

        # Lane assist
        elif "lane" in command_lower:
            if "off" in command_lower or "disable" in command_lower:
                value = False
            elif "on" in command_lower or "enable" in command_lower:
                value = True
            else:
                value = not current_mode["lane_assist"]  # Toggle

            result = ecu.execute_command(
                {"component": "drive_mode", "action": "toggle_lane_assist", "value": value}
            )
            additional_info = f"Lane keeping assist {'enabled' if value else 'disabled'}"

        # Adaptive cruise
        elif "cruise" in command_lower or "adaptive" in command_lower:
            if vehicle_info["speed"] < 25:
                return "Adaptive cruise control requires minimum speed of 25 mph"

            if "off" in command_lower or "disable" in command_lower:
                value = False
            elif "on" in command_lower or "enable" in command_lower:
                value = True
            else:
                value = not current_mode["adaptive_cruise"]  # Toggle

            result = ecu.execute_command(
                {"component": "drive_mode", "action": "toggle_cruise", "value": value}
            )
            additional_info = f"Adaptive cruise control {'engaged' if value else 'disengaged'}"

        else:
            # Show current state
            return f"""Current drive settings:
- Mode: {current_mode['current'].upper()}
- Traction Control: {'On' if current_mode['traction_control'] else 'Off'}
- Stability Control: {'On' if current_mode['stability_control'] else 'Off'}
- Lane Assist: {'On' if current_mode['lane_assist'] else 'Off'}
- Adaptive Cruise: {'On' if current_mode['adaptive_cruise'] else 'Off'}
- Vehicle Speed: {vehicle_info['speed']} mph

Available modes:
- Sport: Enhanced performance, reduced economy
- Eco: Maximum efficiency, gentler acceleration
- Normal: Balanced comfort and performance
- Snow: Optimized for slippery conditions

You can say:
- "Switch to sport mode"
- "Enable eco mode"
- "Turn on traction control" """

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
