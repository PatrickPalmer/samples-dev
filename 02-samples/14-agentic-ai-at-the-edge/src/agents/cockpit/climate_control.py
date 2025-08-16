"""
Climate Control Agent - Manages vehicle HVAC system
Supports TechCar Model X and AutoDrive CX-7
"""

import logging
from strands import tool
from ...data.vehicle_systems import get_virtual_ecu

logger = logging.getLogger(__name__)


@tool
def climate_control(command: str) -> str:
    """
    Control the vehicle's climate system including temperature, fan speed, and air distribution.

    Examples:
    - "Set temperature to 72 degrees"
    - "Turn on max AC"
    - "Increase fan speed"
    - "Turn on defrost"
    - "I'm too hot" (will lower temperature)
    - "It's freezing" (will increase heat)

    Args:
        command: Natural language climate control request

    Returns:
        Confirmation of climate adjustment with current settings
    """
    ecu = get_virtual_ecu()
    current_climate = ecu.get_state("climate")

    # Parse command to determine action
    command_lower = command.lower()

    try:
        # Temperature adjustment
        if any(
            word in command_lower
            for word in ["temperature", "temp", "degrees", "warmer", "cooler", "hot", "cold"]
        ):
            if "set" in command_lower or "to" in command_lower:
                # Extract temperature value
                import re

                temp_match = re.search(r"(\d+)\s*(?:degrees?|°)?", command_lower)
                if temp_match:
                    target_temp = int(temp_match.group(1))
                    result = ecu.execute_command(
                        {"component": "climate", "action": "set_temperature", "value": target_temp}
                    )
                else:
                    return "Please specify a temperature between 60 and 85 degrees"
            elif (
                "warmer" in command_lower
                or "hotter" in command_lower
                or "increase" in command_lower
            ):
                new_temp = min(current_climate["temperature_set"] + 3, 85)
                result = ecu.execute_command(
                    {"component": "climate", "action": "set_temperature", "value": new_temp}
                )
            elif (
                "cooler" in command_lower
                or "colder" in command_lower
                or "decrease" in command_lower
                or "too hot" in command_lower
            ):
                new_temp = max(current_climate["temperature_set"] - 3, 60)
                result = ecu.execute_command(
                    {"component": "climate", "action": "set_temperature", "value": new_temp}
                )
            elif "freezing" in command_lower or "too cold" in command_lower:
                new_temp = min(current_climate["temperature_set"] + 5, 85)
                result = ecu.execute_command(
                    {"component": "climate", "action": "set_temperature", "value": new_temp}
                )
            else:
                # Default temperature adjustment
                result = ecu.execute_command(
                    {"component": "climate", "action": "set_temperature", "value": 72}
                )

        # Max cooling
        elif any(
            phrase in command_lower
            for phrase in ["max ac", "max cool", "maximum cooling", "blast cold"]
        ):
            result = ecu.execute_command({"component": "climate", "action": "max_cool"})

        # Max heating
        elif any(phrase in command_lower for phrase in ["max heat", "maximum heat", "blast heat"]):
            result = ecu.execute_command({"component": "climate", "action": "max_heat"})

        # Defrost
        elif "defrost" in command_lower or "defog" in command_lower:
            result = ecu.execute_command({"component": "climate", "action": "defrost"})

        # Fan speed
        elif "fan" in command_lower:
            if "increase" in command_lower or "higher" in command_lower or "up" in command_lower:
                new_speed = min(current_climate["fan_speed"] + 2, 7)
                result = ecu.execute_command(
                    {"component": "climate", "action": "set_fan_speed", "value": new_speed}
                )
            elif "decrease" in command_lower or "lower" in command_lower or "down" in command_lower:
                new_speed = max(current_climate["fan_speed"] - 2, 0)
                result = ecu.execute_command(
                    {"component": "climate", "action": "set_fan_speed", "value": new_speed}
                )
            elif "max" in command_lower or "full" in command_lower:
                result = ecu.execute_command(
                    {"component": "climate", "action": "set_fan_speed", "value": 7}
                )
            elif "off" in command_lower or "stop" in command_lower:
                result = ecu.execute_command(
                    {"component": "climate", "action": "set_fan_speed", "value": 0}
                )
            else:
                # Try to extract fan speed number
                import re

                speed_match = re.search(r"(\d+)", command_lower)
                if speed_match:
                    speed = min(int(speed_match.group(1)), 7)
                    result = ecu.execute_command(
                        {"component": "climate", "action": "set_fan_speed", "value": speed}
                    )
                else:
                    return f"Current fan speed is {current_climate['fan_speed']}. Please specify a speed from 0-7."

        # AC control
        elif "ac" in command_lower or "air conditioning" in command_lower:
            if "on" in command_lower:
                result = ecu.execute_command(
                    {"component": "climate", "action": "toggle_ac", "value": True}
                )
            elif "off" in command_lower:
                result = ecu.execute_command(
                    {"component": "climate", "action": "toggle_ac", "value": False}
                )
            else:
                # Toggle AC
                result = ecu.execute_command(
                    {
                        "component": "climate",
                        "action": "toggle_ac",
                        "value": not current_climate["ac_on"],
                    }
                )

        # Recirculation
        elif "recirc" in command_lower or "recirculation" in command_lower:
            current_recirc = current_climate.get("recirc", False)
            result = ecu.execute_command(
                {"component": "climate", "action": "toggle_recirc", "value": not current_recirc}
            )

        # Turn off climate
        elif (
            any(phrase in command_lower for phrase in ["turn off", "stop", "disable"])
            and "climate" in command_lower
        ):
            result = ecu.execute_command(
                {"component": "climate", "action": "set_fan_speed", "value": 0}
            )

        else:
            # Default response with current state
            return f"""Current climate settings:
- Temperature: {current_climate['temperature_set']}°F (current cabin: {current_climate['temperature_current']}°F)
- Fan speed: {current_climate['fan_speed']}/7
- Mode: {current_climate['mode']}
- AC: {'On' if current_climate['ac_on'] else 'Off'}

You can say things like:
- "Set temperature to 72"
- "Turn on max AC"
- "Increase fan speed"
- "Turn on defrost" """

        # Process result
        if result["success"]:
            message = result["message"]
            if result.get("estimated_time", 0) > 0:
                minutes = result["estimated_time"] // 60
                if minutes > 0:
                    message += (
                        f" (reaching target in about {minutes} minute{'s' if minutes != 1 else ''})"
                    )

            # Add CAN signal info for demo
            if result.get("can_signals"):
                logger.debug(f"CAN signals: {result['can_signals']}")

            return message
        else:
            return f"Cannot adjust climate: {result['message']}"

    except Exception as e:
        logger.error(f"Climate control error: {e}")
        return f"Sorry, I couldn't process that climate command. Current temperature is set to {current_climate['temperature_set']}°F."
