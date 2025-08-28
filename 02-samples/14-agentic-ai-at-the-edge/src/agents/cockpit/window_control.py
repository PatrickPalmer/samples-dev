"""
Window Control Agent - Manages all vehicle windows and sunroof
Supports TechCar Model X and AutoDrive CX-7
"""

import logging
from strands import tool
from data.vehicle_systems import get_virtual_ecu

logger = logging.getLogger(__name__)


@tool
def window_control(
    action: str,
    target: str,
    position: int = None,
    enable: bool = None
) -> str:
    """
    Control vehicle windows including driver, passenger, rear windows, and sunroof.

    Args:
        action: The window control action to perform. Options:
            - "open": Open window(s) to specified position or fully
            - "close": Close window(s) completely
            - "set_position": Set window(s) to specific position percentage
            - "vent": Open window(s) slightly for ventilation
            - "express_up": Quickly close window(s)
            - "express_down": Quickly open window(s)
            - "toggle_child_lock": Toggle child safety lock
        target: Which window(s) to control. Options:
            - "driver": Driver side window
            - "passenger": Passenger side window
            - "rear_left": Rear left window
            - "rear_right": Rear right window
            - "rear": Both rear windows
            - "all": All windows
            - "sunroof": Sunroof
        position: Window position percentage (0=closed, 100=fully open) for set_position
        enable: Enable/disable for toggle actions like child lock

    Returns:
        Confirmation of window adjustment with safety status
    """
    ecu = get_virtual_ecu()
    current_windows = ecu.get_state("windows")
    vehicle_info = ecu.get_state("vehicle_info")

    try:
        # Validate action parameter
        valid_actions = [
            "open", "close", "set_position", "vent", "express_up", "express_down", "toggle_child_lock"
        ]
        if action not in valid_actions:
            return f"Invalid action '{action}'. Valid actions: {', '.join(valid_actions)}"

        # Validate target parameter
        valid_targets = ["driver", "passenger", "rear_left", "rear_right", "rear", "all", "sunroof"]
        if target not in valid_targets:
            return f"Invalid target '{target}'. Valid targets: {', '.join(valid_targets)}"

        # Handle vent action (special case for all windows)
        if action == "vent":
            if target == "all":
                result = ecu.execute_command({"component": "windows", "action": "vent_all"})
            else:
                # Vent specific window to 15% open
                result = ecu.execute_command({
                    "component": "windows",
                    "action": "set_position",
                    "target": target,
                    "value": 15
                })

        # Handle close action
        elif action == "close":
            if target == "all":
                result = ecu.execute_command({"component": "windows", "action": "close_all"})
            elif target == "rear":
                # Close both rear windows
                result1 = ecu.execute_command({
                    "component": "windows", "action": "set_position", "target": "rear_left", "value": 0
                })
                result = ecu.execute_command({
                    "component": "windows", "action": "set_position", "target": "rear_right", "value": 0
                })
            else:
                result = ecu.execute_command({
                    "component": "windows", "action": "set_position", "target": target, "value": 0
                })

        # Handle express actions
        elif action == "express_up":
            result = ecu.execute_command({"component": "windows", "action": "express_up", "target": target})

        elif action == "express_down":
            result = ecu.execute_command({"component": "windows", "action": "express_down", "target": target})

        # Handle open action (default to fully open)
        elif action == "open":
            open_position = 100  # Default to fully open

            if target == "all":
                # Open all windows to specified position
                for window in ["driver", "passenger", "rear_left", "rear_right"]:
                    result = ecu.execute_command({
                        "component": "windows", "action": "set_position", "target": window, "value": open_position
                    })
            elif target == "rear":
                # Open both rear windows
                result1 = ecu.execute_command({
                    "component": "windows", "action": "set_position", "target": "rear_left", "value": open_position
                })
                result = ecu.execute_command({
                    "component": "windows", "action": "set_position", "target": "rear_right", "value": open_position
                })
            else:
                result = ecu.execute_command({
                    "component": "windows", "action": "set_position", "target": target, "value": open_position
                })

        # Handle set_position action
        elif action == "set_position":
            if position is None:
                return "Position value required for set_position action"
            if not (0 <= position <= 100):
                return "Position must be between 0 and 100 percent"

            if target == "all":
                for window in ["driver", "passenger", "rear_left", "rear_right"]:
                    result = ecu.execute_command({
                        "component": "windows", "action": "set_position", "target": window, "value": position
                    })
            elif target == "rear":
                result1 = ecu.execute_command({
                    "component": "windows", "action": "set_position", "target": "rear_left", "value": position
                })
                result = ecu.execute_command({
                    "component": "windows", "action": "set_position", "target": "rear_right", "value": position
                })
            else:
                result = ecu.execute_command({
                    "component": "windows", "action": "set_position", "target": target, "value": position
                })

        # Handle child lock toggle
        elif action == "toggle_child_lock":
            if enable is None:
                enable = not current_windows.get("child_lock", False)
            result = ecu.execute_command({
                "component": "windows", "action": "toggle_child_lock", "value": enable
            })

        else:
            return f"Unknown action '{action}'. Valid actions: {', '.join(valid_actions)}"

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
