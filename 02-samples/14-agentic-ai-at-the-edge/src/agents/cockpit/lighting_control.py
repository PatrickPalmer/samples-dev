"""
Lighting Control Agent - Manages interior and exterior vehicle lighting
Supports TechCar Model X and AutoDrive CX-7
"""

import logging
from strands import tool
from ...data.vehicle_systems import get_virtual_ecu

logger = logging.getLogger(__name__)


@tool
def lighting_control(command: str) -> str:
    """
    Control vehicle interior and exterior lighting including headlights, ambient lighting, and reading lights.

    Examples:
    - "Turn on headlights"
    - "Set headlights to auto"
    - "Turn on ambient lighting blue"
    - "Dim interior lights"
    - "Turn on reading light"
    - "It's dark in here" (will turn on interior lights)

    Args:
        command: Natural language lighting control request

    Returns:
        Confirmation of lighting adjustment
    """
    ecu = get_virtual_ecu()
    current_lighting = ecu.get_state("lighting")
    vehicle_info = ecu.get_state("vehicle_info")

    command_lower = command.lower()

    try:
        # Headlights control
        if "headlight" in command_lower or "head light" in command_lower:
            if "auto" in command_lower:
                mode = "auto"
            elif "high" in command_lower or "bright" in command_lower:
                mode = "high"
            elif "off" in command_lower:
                mode = "off"
            elif "on" in command_lower:
                mode = "on"
            elif "parking" in command_lower:
                mode = "parking"
            else:
                mode = "on"  # Default to on

            result = ecu.execute_command(
                {"component": "lighting", "action": "set_headlights", "value": mode}
            )

        # Fog lights
        elif "fog" in command_lower:
            if "off" in command_lower:
                value = False
            elif "on" in command_lower:
                value = True
            else:
                value = not current_lighting["fog_lights"]  # Toggle

            result = ecu.execute_command(
                {"component": "lighting", "action": "toggle_fog", "value": value}
            )

        # Ambient lighting
        elif "ambient" in command_lower or "mood" in command_lower:
            # Determine color if specified
            color = "white"  # Default
            intensity = 60  # Default

            colors = ["red", "blue", "green", "white", "orange", "purple", "yellow"]
            for c in colors:
                if c in command_lower:
                    color = c
                    break

            # Check intensity
            if "bright" in command_lower or "max" in command_lower:
                intensity = 100
            elif "dim" in command_lower or "low" in command_lower:
                intensity = 30
            elif "medium" in command_lower or "normal" in command_lower:
                intensity = 60
            else:
                # Try to extract percentage
                import re

                percent_match = re.search(r"(\d+)\s*(?:%|percent)?", command_lower)
                if percent_match:
                    intensity = min(int(percent_match.group(1)), 100)

            if "off" in command_lower:
                result = ecu.execute_command(
                    {
                        "component": "lighting",
                        "action": "set_ambient",
                        "color": color,
                        "intensity": 0,
                    }
                )
            else:
                result = ecu.execute_command(
                    {
                        "component": "lighting",
                        "action": "set_ambient",
                        "color": color,
                        "intensity": intensity,
                    }
                )

        # Interior dome lights
        elif any(word in command_lower for word in ["interior", "dome", "cabin", "dark"]):
            if "off" in command_lower:
                mode = "off"
            elif "on" in command_lower or "dark" in command_lower:
                mode = "on"
            elif "auto" in command_lower:
                mode = "auto"
            elif "dim" in command_lower:
                mode = "dim"
            elif "bright" in command_lower:
                mode = "bright"
            else:
                mode = "on"  # Default to on

            result = ecu.execute_command(
                {"component": "lighting", "action": "set_interior", "value": mode}
            )

        # Reading lights
        elif "reading" in command_lower or "map" in command_lower:
            # Determine which reading light
            if "passenger" in command_lower:
                target = "passenger"
            elif "rear left" in command_lower or "back left" in command_lower:
                target = "rear_left"
            elif "rear right" in command_lower or "back right" in command_lower:
                target = "rear_right"
            elif "rear" in command_lower or "back" in command_lower:
                target = "rear_left"  # Default to rear left
            else:
                target = "driver"  # Default to driver

            if "off" in command_lower:
                value = False
            elif "on" in command_lower:
                value = True
            else:
                value = not current_lighting["reading"].get(target, False)  # Toggle

            result = ecu.execute_command(
                {
                    "component": "lighting",
                    "action": "reading_light",
                    "target": target,
                    "value": value,
                }
            )

        # All lights off
        elif "all" in command_lower and "off" in command_lower:
            # Turn off all interior lights
            result = ecu.execute_command(
                {"component": "lighting", "action": "set_interior", "value": "off"}
            )
            # Also turn off ambient
            ecu.execute_command({"component": "lighting", "action": "set_ambient", "intensity": 0})

        # All lights on
        elif "all" in command_lower and "on" in command_lower:
            result = ecu.execute_command(
                {"component": "lighting", "action": "set_interior", "value": "on"}
            )

        else:
            # Show current state
            ambient = current_lighting["ambient"]
            reading = current_lighting["reading"]

            return f"""Current lighting settings:
- Headlights: {current_lighting['headlights']}
- Fog lights: {'On' if current_lighting['fog_lights'] else 'Off'}
- Interior: {current_lighting['interior_dome']}
- Ambient: {'On' if ambient['enabled'] else 'Off'} ({ambient['color']}, {ambient['intensity']}%)
- Reading lights: Driver: {'On' if reading['driver'] else 'Off'}, Passenger: {'On' if reading['passenger'] else 'Off'}

You can say:
- "Turn on headlights"
- "Set ambient lighting to blue"
- "Turn on reading light"
- "Set headlights to auto" """

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
