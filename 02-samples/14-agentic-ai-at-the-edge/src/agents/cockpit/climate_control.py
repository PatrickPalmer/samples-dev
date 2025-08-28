"""
Climate Control Agent - Manages vehicle HVAC system
Supports TechCar Model X and AutoDrive CX-7
"""

import logging
from strands import tool
from data.vehicle_systems import get_virtual_ecu

logger = logging.getLogger(__name__)


@tool
def climate_control(
    action: str,
    temperature: int = None,
    temperature_adjustment: int = None,
    fan_speed: int = None,
    fan_adjustment: int = None,
    mode: str = None,
    enable: bool = None
) -> str:
    """
    Control the vehicle's climate system including temperature, fan speed, and air distribution.

    Args:
        action: The climate control action to perform. Options:
            - "set_temperature": Set specific temperature
            - "adjust_temperature": Adjust temperature by amount
            - "set_fan_speed": Set specific fan speed
            - "adjust_fan_speed": Adjust fan speed by amount
            - "set_mode": Set climate mode
            - "toggle_ac": Toggle air conditioning
            - "toggle_defrost": Toggle defrost mode
            - "turn_off": Turn off climate system
        temperature: Target temperature in Fahrenheit (60-85) for set_temperature
        temperature_adjustment: Temperature change in degrees (-10 to +10) for adjust_temperature
        fan_speed: Fan speed level (0-7) for set_fan_speed
        fan_adjustment: Fan speed change (-3 to +3) for adjust_fan_speed
        mode: Climate mode for set_mode ("auto", "heat", "cool", "defrost", "vent")
        enable: Enable/disable for toggle actions

    Returns:
        Confirmation of climate adjustment with current settings
    """
    ecu = get_virtual_ecu()
    current_climate = ecu.get_state("climate")

    try:
        # Validate action parameter
        valid_actions = [
            "set_temperature", "adjust_temperature", "set_fan_speed", "adjust_fan_speed",
            "set_mode", "toggle_ac", "toggle_defrost", "turn_off"
        ]
        if action not in valid_actions:
            return f"Invalid action '{action}'. Valid actions: {', '.join(valid_actions)}"

        # Handle temperature actions
        if action == "set_temperature":
            if temperature is None:
                return "Temperature value required for set_temperature action"
            if not (60 <= temperature <= 85):
                return "Temperature must be between 60 and 85 degrees Fahrenheit"
            result = ecu.execute_command(
                {"component": "climate", "action": "set_temperature", "value": temperature}
            )

        elif action == "adjust_temperature":
            if temperature_adjustment is None:
                return "Temperature adjustment value required for adjust_temperature action"
            if not (-10 <= temperature_adjustment <= 10):
                return "Temperature adjustment must be between -10 and +10 degrees"
            new_temp = current_climate["temperature_set"] + temperature_adjustment
            new_temp = max(60, min(85, new_temp))  # Clamp to valid range
            result = ecu.execute_command(
                {"component": "climate", "action": "set_temperature", "value": new_temp}
            )

        # Handle fan speed actions
        elif action == "set_fan_speed":
            if fan_speed is None:
                return "Fan speed value required for set_fan_speed action"
            if not (0 <= fan_speed <= 7):
                return "Fan speed must be between 0 and 7"
            result = ecu.execute_command(
                {"component": "climate", "action": "set_fan_speed", "value": fan_speed}
            )

        elif action == "adjust_fan_speed":
            if fan_adjustment is None:
                return "Fan adjustment value required for adjust_fan_speed action"
            if not (-3 <= fan_adjustment <= 3):
                return "Fan adjustment must be between -3 and +3"
            new_speed = current_climate["fan_speed"] + fan_adjustment
            new_speed = max(0, min(7, new_speed))  # Clamp to valid range
            result = ecu.execute_command(
                {"component": "climate", "action": "set_fan_speed", "value": new_speed}
            )

        # Handle mode actions
        elif action == "set_mode":
            if mode is None:
                return "Mode value required for set_mode action"
            valid_modes = ["auto", "heat", "cool", "defrost", "vent"]
            if mode not in valid_modes:
                return f"Invalid mode '{mode}'. Valid modes: {', '.join(valid_modes)}"

            if mode == "defrost":
                result = ecu.execute_command({"component": "climate", "action": "defrost"})
            elif mode == "cool":
                result = ecu.execute_command({"component": "climate", "action": "max_cool"})
            elif mode == "heat":
                result = ecu.execute_command({"component": "climate", "action": "max_heat"})
            else:
                result = ecu.execute_command(
                    {"component": "climate", "action": "set_mode", "value": mode}
                )

        # Handle toggle actions
        elif action == "toggle_ac":
            if enable is None:
                # Toggle current state
                enable = not current_climate["ac_on"]
            result = ecu.execute_command(
                {"component": "climate", "action": "toggle_ac", "value": enable}
            )

        elif action == "toggle_defrost":
            if enable is None:
                return "Enable value required for toggle_defrost action"
            result = ecu.execute_command({"component": "climate", "action": "defrost"})

        elif action == "turn_off":
            result = ecu.execute_command(
                {"component": "climate", "action": "set_fan_speed", "value": 0}
            )

        else:
            return f"Unknown action '{action}'. Valid actions: {', '.join(valid_actions)}"

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
