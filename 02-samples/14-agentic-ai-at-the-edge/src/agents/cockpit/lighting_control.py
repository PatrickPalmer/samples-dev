"""
Lighting Control Agent - Manages interior and exterior vehicle lighting
Supports TechCar Model X and AutoDrive CX-7
"""

import logging
from strands import tool
from data.vehicle_systems import get_virtual_ecu

logger = logging.getLogger(__name__)


@tool
def lighting_control(
    action: str,
    headlight_mode: str = None,
    interior_brightness: int = None,
    ambient_color: str = None,
    ambient_brightness: int = None,
    reading_light: str = None,
    enable: bool = None
) -> str:
    """
    Control vehicle interior and exterior lighting including headlights, ambient lighting, and reading lights.

    Args:
        action: The lighting control action to perform. Options:
            - "set_headlights": Set headlight mode
            - "set_interior": Set interior light brightness
            - "set_ambient": Set ambient lighting color and brightness
            - "set_reading": Control reading lights
            - "toggle_auto": Toggle automatic lighting
        headlight_mode: Headlight mode for set_headlights ("off", "on", "auto", "high_beam")
        interior_brightness: Interior light brightness percentage (0-100) for set_interior
        ambient_color: Ambient lighting color for set_ambient ("white", "blue", "red", "green", "purple", "orange")
        ambient_brightness: Ambient lighting brightness percentage (0-100) for set_ambient
        reading_light: Which reading light to control for set_reading ("driver", "passenger", "rear_left", "rear_right", "all")
        enable: Enable/disable for toggle actions

    Returns:
        Confirmation of lighting adjustment
    """
    ecu = get_virtual_ecu()
    current_lighting = ecu.get_state("lighting")
    vehicle_info = ecu.get_state("vehicle_info")

    try:
        # Validate action parameter
        valid_actions = ["set_headlights", "set_interior", "set_ambient", "set_reading", "toggle_auto"]
        if action not in valid_actions:
            return f"Invalid action '{action}'. Valid actions: {', '.join(valid_actions)}"

        # Handle headlight control
        if action == "set_headlights":
            if headlight_mode is None:
                return "Headlight mode required for set_headlights action"
            valid_modes = ["off", "on", "auto", "high_beam"]
            if headlight_mode not in valid_modes:
                return f"Invalid headlight mode '{headlight_mode}'. Valid modes: {', '.join(valid_modes)}"

            # Map high_beam to the ECU's expected value
            ecu_mode = "high" if headlight_mode == "high_beam" else headlight_mode
            result = ecu.execute_command({
                "component": "lighting", "action": "set_headlights", "value": ecu_mode
            })

        # Handle interior lighting
        elif action == "set_interior":
            if interior_brightness is None:
                return "Interior brightness required for set_interior action"
            if not (0 <= interior_brightness <= 100):
                return "Interior brightness must be between 0 and 100 percent"
            result = ecu.execute_command({
                "component": "lighting", "action": "set_interior", "value": interior_brightness
            })

        # Handle ambient lighting
        elif action == "set_ambient":
            if ambient_color is None:
                return "Ambient color required for set_ambient action"
            valid_colors = ["white", "blue", "red", "green", "purple", "orange"]
            if ambient_color not in valid_colors:
                return f"Invalid ambient color '{ambient_color}'. Valid colors: {', '.join(valid_colors)}"

            # Use provided brightness or default to 60%
            brightness = ambient_brightness if ambient_brightness is not None else 60
            if not (0 <= brightness <= 100):
                return "Ambient brightness must be between 0 and 100 percent"

            result = ecu.execute_command({
                "component": "lighting",
                "action": "set_ambient",
                "color": ambient_color,
                "intensity": brightness
            })
        # Handle reading lights
        elif action == "set_reading":
            if reading_light is None:
                return "Reading light target required for set_reading action"
            valid_targets = ["driver", "passenger", "rear_left", "rear_right", "all"]
            if reading_light not in valid_targets:
                return f"Invalid reading light target '{reading_light}'. Valid targets: {', '.join(valid_targets)}"

            # Use enable parameter or default to toggle
            if enable is None:
                if reading_light == "all":
                    # For "all", turn on if any are off, otherwise turn off
                    any_on = any(current_lighting["reading"].values())
                    enable = not any_on
                else:
                    enable = not current_lighting["reading"].get(reading_light, False)

            if reading_light == "all":
                # Control all reading lights
                for target in ["driver", "passenger", "rear_left", "rear_right"]:
                    result = ecu.execute_command({
                        "component": "lighting", "action": "toggle_reading", "target": target, "value": enable
                    })
            else:
                result = ecu.execute_command({
                    "component": "lighting", "action": "toggle_reading", "target": reading_light, "value": enable
                })

        # Handle auto toggle
        elif action == "toggle_auto":
            if enable is None:
                enable = not current_lighting.get("auto_mode", False)
            result = ecu.execute_command({
                "component": "lighting", "action": "toggle_auto", "value": enable
            })
        else:
            return f"Unknown action '{action}'. Valid actions: {', '.join(valid_actions)}"

        # Process result
        if result["success"]:
            message = result["message"]

            # Add vehicle state context
            if vehicle_info["speed"] > 0 and "headlight" in command_lower:
                message += f" (Vehicle moving at {vehicle_info['speed']} mph)"

            # Log CAN signals
            if result.get("can_signals"):
                logger.debug(f"CAN signals: {result['can_signals']}")

            return message
        else:
            return f"Cannot adjust lighting: {result['message']}"

    except Exception as e:
        logger.error(f"Lighting control error: {e}")
        return "Sorry, I couldn't process that lighting command. Please specify what lights to control (headlights, interior, ambient, reading)."
