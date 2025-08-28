"""
Seat Control Agent - Manages seat positions and comfort features
Supports TechCar Model X and AutoDrive CX-7
"""

import logging
from strands import tool
from data.vehicle_systems import get_virtual_ecu

logger = logging.getLogger(__name__)


@tool
def seat_control(
    action: str,
    seat: str,
    position_type: str = None,
    adjustment: int = None,
    heating_level: int = None,
    cooling_level: int = None,
    lumbar_adjustment: int = None,
    memory_slot: int = None
) -> str:
    """
    Control vehicle seat position, heating, cooling, and memory settings.
    Safety: Only works when vehicle is parked or moving below 5 mph.

    Args:
        action: The seat control action to perform. Options:
            - "adjust_position": Adjust seat position (forward/back, height, tilt)
            - "set_heating": Set seat heating level
            - "set_cooling": Set seat cooling level
            - "adjust_lumbar": Adjust lumbar support
            - "save_memory": Save current position to memory slot
            - "recall_memory": Recall position from memory slot
        seat: Which seat to control ("driver" or "passenger")
        position_type: Type of position adjustment ("forward", "height", "tilt") for adjust_position
        adjustment: Position adjustment amount (-20 to +20) for adjust_position
        heating_level: Seat heating level (0-3, 0=off) for set_heating
        cooling_level: Seat cooling level (0-3, 0=off) for set_cooling
        lumbar_adjustment: Lumbar support adjustment (-5 to +5) for adjust_lumbar
        memory_slot: Memory slot number (1-3) for save_memory/recall_memory

    Returns:
        Confirmation of seat adjustment or safety warning
    """
    ecu = get_virtual_ecu()
    vehicle_info = ecu.get_state("vehicle_info")
    current_seats = ecu.get_state("seats")

    # Safety check first
    if vehicle_info["speed"] > 5:
        return f"⚠️ Safety: Cannot adjust seat while vehicle is moving at {vehicle_info['speed']} mph. Please stop or slow down below 5 mph."

    try:
        # Validate action parameter
        valid_actions = [
            "adjust_position", "set_heating", "set_cooling", "adjust_lumbar", "save_memory", "recall_memory"
        ]
        if action not in valid_actions:
            return f"Invalid action '{action}'. Valid actions: {', '.join(valid_actions)}"

        # Validate seat parameter
        valid_seats = ["driver", "passenger"]
        if seat not in valid_seats:
            return f"Invalid seat '{seat}'. Valid seats: {', '.join(valid_seats)}"

        current_seat = current_seats[seat]

        # Handle position adjustments
        if action == "adjust_position":
            if position_type is None:
                return "Position type required for adjust_position action (forward, height, tilt)"
            if adjustment is None:
                return "Adjustment value required for adjust_position action"

            valid_position_types = ["forward", "height", "tilt"]
            if position_type not in valid_position_types:
                return f"Invalid position type '{position_type}'. Valid types: {', '.join(valid_position_types)}"

            if not (-20 <= adjustment <= 20):
                return "Adjustment must be between -20 and +20"

            if position_type == "forward":
                new_position = current_seat["position_forward"] + adjustment
                new_position = max(0, min(100, new_position))  # Clamp to valid range
                result = ecu.execute_command({
                    "component": "seats", "action": "adjust_position", "seat": seat, "forward": new_position
                })
            elif position_type == "height":
                new_height = current_seat["position_height"] + adjustment
                new_height = max(0, min(100, new_height))  # Clamp to valid range
                result = ecu.execute_command({
                    "component": "seats", "action": "adjust_position", "seat": seat, "height": new_height
                })
            elif position_type == "tilt":
                new_tilt = current_seat["position_recline"] + adjustment
                new_tilt = max(0, min(60, new_tilt))  # Clamp to valid range
                result = ecu.execute_command({
                    "component": "seats", "action": "adjust_position", "seat": seat, "recline": new_tilt
                })

        # Handle heating control
        elif action == "set_heating":
            if heating_level is None:
                return "Heating level required for set_heating action"
            if not (0 <= heating_level <= 3):
                return "Heating level must be between 0 (off) and 3 (high)"
            result = ecu.execute_command({
                "component": "seats", "action": "set_heating", "seat": seat, "value": heating_level
            })

        # Handle cooling control
        elif action == "set_cooling":
            if cooling_level is None:
                return "Cooling level required for set_cooling action"
            if not (0 <= cooling_level <= 3):
                return "Cooling level must be between 0 (off) and 3 (high)"
            result = ecu.execute_command({
                "component": "seats", "action": "set_cooling", "seat": seat, "value": cooling_level
            })

        # Handle lumbar support adjustment
        elif action == "adjust_lumbar":
            if lumbar_adjustment is None:
                return "Lumbar adjustment value required for adjust_lumbar action"
            if not (-5 <= lumbar_adjustment <= 5):
                return "Lumbar adjustment must be between -5 and +5"
            new_lumbar = current_seat["lumbar"] + lumbar_adjustment
            new_lumbar = max(1, min(5, new_lumbar))  # Clamp to valid range
            result = ecu.execute_command({
                "component": "seats", "action": "set_lumbar", "seat": seat, "value": new_lumbar
            })
        # Handle memory save
        elif action == "save_memory":
            if memory_slot is None:
                return "Memory slot required for save_memory action"
            if not (1 <= memory_slot <= 3):
                return "Memory slot must be between 1 and 3"
            result = ecu.execute_command({
                "component": "seats", "action": "memory_save", "seat": seat, "position": memory_slot
            })

        # Handle memory recall
        elif action == "recall_memory":
            if memory_slot is None:
                return "Memory slot required for recall_memory action"
            if not (1 <= memory_slot <= 3):
                return "Memory slot must be between 1 and 3"
            result = ecu.execute_command({
                "component": "seats", "action": "memory_recall", "seat": seat, "position": memory_slot
            })

        else:
            return f"Unknown action '{action}'. Valid actions: {', '.join(valid_actions)}"

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
