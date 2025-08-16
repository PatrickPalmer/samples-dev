"""
Seat Control Agent - Manages seat positions and comfort features
Supports TechCar Model X and AutoDrive CX-7
"""

import logging
from strands import tool
from ...data.vehicle_systems import get_virtual_ecu

logger = logging.getLogger(__name__)


@tool
def seat_control(command: str) -> str:
    """
    Control vehicle seat position, heating, cooling, and memory settings.
    Safety: Only works when vehicle is parked or moving below 5 mph.

    Examples:
    - "Move seat back"
    - "Turn on seat heating"
    - "Save seat position as memory 1"
    - "Recall seat position 2"
    - "Adjust lumbar support"
    - "My back hurts" (will adjust lumbar)

    Args:
        command: Natural language seat control request

    Returns:
        Confirmation of seat adjustment or safety warning
    """
    ecu = get_virtual_ecu()
    vehicle_info = ecu.get_state("vehicle_info")
    current_seats = ecu.get_state("seats")

    # Safety check first
    if vehicle_info["speed"] > 5:
        return f"⚠️ Safety: Cannot adjust seat while vehicle is moving at {vehicle_info['speed']} mph. Please stop or slow down below 5 mph."

    command_lower = command.lower()

    try:
        # Determine which seat
        if "passenger" in command_lower:
            seat = "passenger"
        else:
            seat = "driver"  # Default to driver seat

        current_seat = current_seats[seat]

        # Seat position adjustments
        if any(word in command_lower for word in ["forward", "forwards", "closer", "front"]):
            new_position = min(current_seat["position_forward"] + 10, 100)
            result = ecu.execute_command(
                {
                    "component": "seats",
                    "action": "adjust_position",
                    "seat": seat,
                    "forward": new_position,
                }
            )

        elif any(
            word in command_lower for word in ["back", "backward", "backwards", "rear", "farther"]
        ):
            new_position = max(current_seat["position_forward"] - 10, 0)
            result = ecu.execute_command(
                {
                    "component": "seats",
                    "action": "adjust_position",
                    "seat": seat,
                    "forward": new_position,
                }
            )

        elif any(word in command_lower for word in ["up", "higher", "raise"]):
            new_height = min(current_seat["position_height"] + 10, 100)
            result = ecu.execute_command(
                {
                    "component": "seats",
                    "action": "adjust_position",
                    "seat": seat,
                    "height": new_height,
                }
            )

        elif any(word in command_lower for word in ["down", "lower"]):
            new_height = max(current_seat["position_height"] - 10, 0)
            result = ecu.execute_command(
                {
                    "component": "seats",
                    "action": "adjust_position",
                    "seat": seat,
                    "height": new_height,
                }
            )

        elif "recline" in command_lower:
            if "more" in command_lower:
                new_recline = min(current_seat["position_recline"] + 10, 60)
            elif "less" in command_lower:
                new_recline = max(current_seat["position_recline"] - 10, 0)
            else:
                new_recline = 40  # Comfortable recline position

            result = ecu.execute_command(
                {
                    "component": "seats",
                    "action": "adjust_position",
                    "seat": seat,
                    "recline": new_recline,
                }
            )

        # Lumbar support
        elif (
            "lumbar" in command_lower
            or "back support" in command_lower
            or "back hurts" in command_lower
        ):
            if "increase" in command_lower or "more" in command_lower or "hurts" in command_lower:
                new_lumbar = min(current_seat["lumbar"] + 1, 5)
            elif "decrease" in command_lower or "less" in command_lower:
                new_lumbar = max(current_seat["lumbar"] - 1, 1)
            else:
                new_lumbar = 3  # Medium support

            result = ecu.execute_command(
                {"component": "seats", "action": "set_lumbar", "seat": seat, "value": new_lumbar}
            )

        # Heating
        elif any(word in command_lower for word in ["heat", "heating", "warm", "warmer", "cold"]):
            if "off" in command_lower:
                level = 0
            elif "max" in command_lower or "high" in command_lower:
                level = 3
            elif "medium" in command_lower or "mid" in command_lower:
                level = 2
            elif "low" in command_lower:
                level = 1
            elif "cold" in command_lower or "warm" in command_lower:
                level = 2  # Default to medium
            else:
                # Toggle or increase
                level = min(current_seat["heating"] + 1, 3) if current_seat["heating"] < 3 else 0

            result = ecu.execute_command(
                {"component": "seats", "action": "set_heating", "seat": seat, "value": level}
            )

        # Cooling
        elif any(word in command_lower for word in ["cool", "cooling", "ventilation", "hot"]):
            if "off" in command_lower:
                level = 0
            elif "max" in command_lower or "high" in command_lower:
                level = 3
            elif "medium" in command_lower or "mid" in command_lower:
                level = 2
            elif "low" in command_lower:
                level = 1
            elif "hot" in command_lower:
                level = 2  # Default to medium cooling
            else:
                # Toggle or increase
                level = min(current_seat["cooling"] + 1, 3) if current_seat["cooling"] < 3 else 0

            result = ecu.execute_command(
                {"component": "seats", "action": "set_cooling", "seat": seat, "value": level}
            )

        # Memory positions
        elif "save" in command_lower or "store" in command_lower:
            # Extract position number
            import re

            pos_match = re.search(r"(?:memory|position)?\s*(\d)", command_lower)
            if pos_match:
                position = int(pos_match.group(1))
                if 1 <= position <= 3:
                    result = ecu.execute_command(
                        {
                            "component": "seats",
                            "action": "memory_save",
                            "seat": seat,
                            "position": position,
                        }
                    )
                else:
                    return "Please specify memory position 1, 2, or 3"
            else:
                return "Please specify which memory position to save (1, 2, or 3)"

        elif "recall" in command_lower or "load" in command_lower or "memory" in command_lower:
            # Extract position number
            import re

            pos_match = re.search(r"(?:memory|position)?\s*(\d)", command_lower)
            if pos_match:
                position = int(pos_match.group(1))
                if 1 <= position <= 3:
                    result = ecu.execute_command(
                        {
                            "component": "seats",
                            "action": "memory_recall",
                            "seat": seat,
                            "position": position,
                        }
                    )
                else:
                    return "Please specify memory position 1, 2, or 3"
            else:
                return "Please specify which memory position to recall (1, 2, or 3)"

        else:
            # Show current state
            return f"""{seat.capitalize()} seat settings:
- Position: {current_seat['position_forward']}% forward, {current_seat['position_height']}% height
- Recline: {current_seat['position_recline']}°
- Lumbar support: Level {current_seat['lumbar']}/5
- Heating: Level {current_seat['heating']}/3
- Cooling: Level {current_seat['cooling']}/3

You can say:
- "Move seat back"
- "Turn on seat heating"
- "Save as memory 1"
- "Increase lumbar support" """

        # Process result
        if result["success"]:
            message = result["message"]

            # Add time estimate
            if result.get("estimated_time", 0) > 0:
                message += f" ({result['estimated_time']} seconds)"

            # Log CAN signals
            if result.get("can_signals"):
                logger.debug(f"CAN signals: {result['can_signals']}")

            return message
        else:
            return f"Cannot adjust seat: {result['message']}"

    except Exception as e:
        logger.error(f"Seat control error: {e}")
        return f"Sorry, I couldn't process that seat command. Vehicle must be parked (currently at {vehicle_info['speed']} mph)."
