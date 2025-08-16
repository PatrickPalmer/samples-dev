"""
Window Control Agent - Manages all vehicle windows and sunroof
Supports TechCar Model X and AutoDrive CX-7
"""

import logging
from strands import tool
from ...data.vehicle_systems import get_virtual_ecu

logger = logging.getLogger(__name__)


@tool
def window_control(command: str) -> str:
    """
    Control vehicle windows including driver, passenger, rear windows, and sunroof.

    Examples:
    - "Open driver window"
    - "Close all windows"
    - "Open window halfway"
    - "Vent all windows"
    - "Open sunroof"
    - "It's stuffy in here" (will vent windows)

    Args:
        command: Natural language window control request

    Returns:
        Confirmation of window adjustment with safety status
    """
    ecu = get_virtual_ecu()
    current_windows = ecu.get_state("windows")
    vehicle_info = ecu.get_state("vehicle_info")

    command_lower = command.lower()

    try:
        # Determine target window
        if "driver" in command_lower:
            target = "driver"
        elif "passenger" in command_lower:
            target = "passenger"
        elif "rear left" in command_lower or "back left" in command_lower:
            target = "rear_left"
        elif "rear right" in command_lower or "back right" in command_lower:
            target = "rear_right"
        elif "rear" in command_lower or "back" in command_lower:
            target = "rear"  # Both rear windows
        elif "sunroof" in command_lower or "sun roof" in command_lower:
            target = "sunroof"
        elif "all" in command_lower:
            target = "all"
        else:
            target = "driver"  # Default to driver window

        # Determine action
        if "vent" in command_lower or "crack" in command_lower or "stuffy" in command_lower:
            # Vent windows for fresh air
            result = ecu.execute_command({"component": "windows", "action": "vent_all"})

        elif "close" in command_lower or "up" in command_lower:
            if target == "all":
                result = ecu.execute_command({"component": "windows", "action": "close_all"})
            elif target == "rear":
                # Close both rear windows
                result1 = ecu.execute_command(
                    {
                        "component": "windows",
                        "action": "set_position",
                        "target": "rear_left",
                        "value": 0,
                    }
                )
                result = ecu.execute_command(
                    {
                        "component": "windows",
                        "action": "set_position",
                        "target": "rear_right",
                        "value": 0,
                    }
                )
            else:
                result = ecu.execute_command(
                    {"component": "windows", "action": "express_up", "target": target}
                )

        elif "open" in command_lower or "down" in command_lower:
            # Check for position specification
            position = 100  # Default to fully open

            if "halfway" in command_lower or "half way" in command_lower or "half" in command_lower:
                position = 50
            elif "quarter" in command_lower or "25" in command_lower:
                position = 25
            elif "three quarter" in command_lower or "75" in command_lower:
                position = 75
            elif "little" in command_lower or "bit" in command_lower or "slightly" in command_lower:
                position = 20
            else:
                # Try to extract percentage
                import re

                percent_match = re.search(r"(\d+)\s*(?:%|percent)?", command_lower)
                if percent_match:
                    position = min(int(percent_match.group(1)), 100)

            if target == "all":
                # Open all windows to specified position
                for window in ["driver", "passenger", "rear_left", "rear_right"]:
                    result = ecu.execute_command(
                        {
                            "component": "windows",
                            "action": "set_position",
                            "target": window,
                            "value": position,
                        }
                    )
            elif target == "rear":
                # Open both rear windows
                result1 = ecu.execute_command(
                    {
                        "component": "windows",
                        "action": "set_position",
                        "target": "rear_left",
                        "value": position,
                    }
                )
                result = ecu.execute_command(
                    {
                        "component": "windows",
                        "action": "set_position",
                        "target": "rear_right",
                        "value": position,
                    }
                )
            else:
                if position == 100:
                    result = ecu.execute_command(
                        {"component": "windows", "action": "express_down", "target": target}
                    )
                else:
                    result = ecu.execute_command(
                        {
                            "component": "windows",
                            "action": "set_position",
                            "target": target,
                            "value": position,
                        }
                    )

        elif "stop" in command_lower:
            # Stop current window operation (return current state)
            return f"Window operation stopped. Current positions: Driver: {current_windows['driver']}%, Passenger: {current_windows['passenger']}%"

        else:
            # Show current state
            return f"""Current window positions:
- Driver: {current_windows['driver']}% open
- Passenger: {current_windows['passenger']}% open
- Rear left: {current_windows['rear_left']}% open
- Rear right: {current_windows['rear_right']}% open
- Sunroof: {current_windows['sunroof']}% open
- Child lock: {'Active' if current_windows['child_lock'] else 'Inactive'}

Vehicle speed: {vehicle_info['speed']} mph

You can say:
- "Open driver window"
- "Close all windows"
- "Vent all windows"
- "Open sunroof halfway" """

        # Process result
        if result["success"]:
            message = result["message"]

            # Add safety warnings if applicable
            if vehicle_info["speed"] > 45:
                message += " (Speed limited due to vehicle moving at high speed)"
            elif vehicle_info["speed"] > 0:
                message += f" (Vehicle in motion at {vehicle_info['speed']} mph)"

            # Add time estimate
            if result.get("estimated_time", 0) > 0:
                message += f" - {result['estimated_time']:.1f} seconds"

            # Log CAN signals for demo
            if result.get("can_signals"):
                logger.debug(f"CAN signals: {result['can_signals']}")

            return message
        else:
            # Handle safety violations
            if "SAFETY" in result.get("error", ""):
                if vehicle_info["speed"] > 45:
                    return f"Cannot fully open windows at {vehicle_info['speed']} mph. Maximum opening is 50% for safety."
                elif current_windows.get("child_lock") and "rear" in target:
                    return "Cannot control rear windows - child lock is active. Disable child lock first."
                else:
                    return f"Safety restriction: {result['message']}"
            else:
                return f"Cannot control windows: {result['message']}"

    except Exception as e:
        logger.error(f"Window control error: {e}")
        return "Sorry, I couldn't process that window command. Please try specifying which window and whether to open or close it."
