"""
Virtual ECU Service - Simulates vehicle Electronic Control Unit
For TechCar Model X and AutoDrive CX-7 demo vehicles
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VehicleState(Enum):
    """Vehicle operational states"""

    PARKED = "parked"
    IDLE = "idle"
    DRIVING = "driving"
    CHARGING = "charging"


class SafetyLevel(Enum):
    """Command safety classifications"""

    ALWAYS_SAFE = 1  # Can execute anytime
    PARKED_ONLY = 2  # Only when parked
    LOW_SPEED = 3  # Only below 5 mph
    CONDITIONAL = 4  # Depends on multiple factors


class VirtualECU:
    """
    Mock Electronic Control Unit - Vehicle State Manager
    Simulates realistic vehicle behavior for TechCar Model X and AutoDrive CX-7
    """

    def __init__(self, vehicle_model: str = "TechCar Model X"):
        """Initialize with vehicle model"""
        self.vehicle_model = vehicle_model
        self.state = self._load_initial_state()
        self.state_lock = threading.Lock()
        self.signal_bus = []
        self.state_history = []
        self.active_timers = {}
        logger.info(f"Virtual ECU initialized for {vehicle_model}")

    def _load_initial_state(self) -> Dict:
        """Load default vehicle state based on model"""
        base_state = {
            "vehicle_info": {
                "make": "TechCar" if "TechCar" in self.vehicle_model else "AutoDrive",
                "model": self.vehicle_model,
                "year": 2024,
                "odometer": 15234,
                "fuel_level": 75,
                "battery_voltage": 12.6,
                "engine_on": False,
                "speed": 0,
                "gear": "P",
                "doors_locked": True,
                "parking_brake": True,
            },
            "climate": {
                "temperature_set": 72,
                "temperature_current": 75,
                "temperature_outside": 65,
                "fan_speed": 0,
                "mode": "off",
                "ac_on": False,
                "heater_on": False,
                "recirc": False,
                "defrost_front": False,
                "defrost_rear": False,
            },
            "windows": {
                "driver": 0,
                "passenger": 0,
                "rear_left": 0,
                "rear_right": 0,
                "sunroof": 0,
                "child_lock": False,
            },
            "seats": {
                "driver": {
                    "position_forward": 50,
                    "position_height": 50,
                    "position_recline": 30,
                    "lumbar": 3,
                    "heating": 0,
                    "cooling": 0,
                    "occupied": False,
                    "seatbelt": False,
                },
                "passenger": {
                    "position_forward": 50,
                    "position_height": 50,
                    "position_recline": 30,
                    "lumbar": 3,
                    "heating": 0,
                    "cooling": 0,
                    "occupied": False,
                    "seatbelt": False,
                },
            },
            "lighting": {
                "headlights": "off",
                "fog_lights": False,
                "interior_dome": "off",
                "ambient": {"enabled": False, "color": "white", "intensity": 50},
                "reading": {
                    "driver": False,
                    "passenger": False,
                    "rear_left": False,
                    "rear_right": False,
                },
            },
            "drive_mode": {
                "current": "normal",
                "traction_control": True,
                "stability_control": True,
                "lane_assist": True,
                "adaptive_cruise": False,
            },
        }

        # Add model-specific features
        if "AutoDrive" in self.vehicle_model:
            base_state["seats"]["driver"]["massage"] = False
            base_state["drive_mode"]["air_suspension"] = "normal"

        return base_state

    def execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for all vehicle commands
        """
        with self.state_lock:
            # Log command
            logger.debug(f"ECU Command: {json.dumps(command)}")

            # Validate safety
            safety_check = self._check_safety(command)
            if not safety_check["safe"]:
                return {
                    "success": False,
                    "error": safety_check["error_code"],
                    "message": safety_check["message"],
                    "current_state": self._get_component_state(command["component"]),
                }

            # Validate constraints
            constraint_check = self._validate_constraints(command)
            if not constraint_check["valid"]:
                return {
                    "success": False,
                    "error": "CONSTRAINT_VIOLATION",
                    "message": constraint_check["message"],
                    "current_state": self._get_component_state(command["component"]),
                }

            # Execute command
            result = self._execute_component_command(command)

            # Generate CAN signals
            result["can_signals"] = self._generate_can_signals(command, result)

            # Add to history
            self._add_to_history(command, result)

            return result

    def get_state(self, component: Optional[str] = None) -> Dict:
        """Get current state of component or entire vehicle"""
        with self.state_lock:
            if component:
                return self.state.get(component, {}).copy()
            return self.state.copy()

    def _check_safety(self, command: Dict) -> Dict[str, Any]:
        """Check if command is safe to execute"""
        component = command["component"]
        action = command["action"]

        # Safety rules
        safety_rules = {
            "seats": {
                "check": lambda: self.state["vehicle_info"]["speed"] == 0,
                "message": "Cannot adjust seat while vehicle is moving",
            },
            "windows": {
                "check": lambda: self._check_window_safety(command),
                "message": "Window operation restricted at current speed",
            },
            "drive_mode": {
                "check": lambda: self.state["vehicle_info"]["speed"] < 5,
                "message": "Cannot change drive mode above 5 mph",
            },
        }

        if component in safety_rules:
            rule = safety_rules[component]
            if not rule["check"]():
                return {
                    "safe": False,
                    "error_code": f"SAFETY_{component.upper()}",
                    "message": rule["message"],
                }

        return {"safe": True}

    def _check_window_safety(self, command: Dict) -> bool:
        """Special safety logic for windows"""
        speed = self.state["vehicle_info"]["speed"]

        # No full open above 45 mph
        if speed > 45 and command.get("value", 0) > 50:
            return False

        # Child lock check
        if self.state["windows"]["child_lock"] and command.get("target") in [
            "rear_left",
            "rear_right",
        ]:
            return False

        return True

    def _validate_constraints(self, command: Dict) -> Dict[str, Any]:
        """Validate command parameters against constraints"""
        component = command["component"]

        constraints = {
            "climate": {"temperature": (60, 85), "fan_speed": (0, 7)},
            "windows": {"position": (0, 100)},
            "seats": {
                "position_forward": (0, 100),
                "position_height": (0, 100),
                "position_recline": (0, 60),
                "heating": (0, 3),
                "cooling": (0, 3),
            },
            "lighting": {"intensity": (0, 100)},
        }

        if component in constraints:
            comp_constraints = constraints[component]

            for param, (min_val, max_val) in comp_constraints.items():
                if param in command:
                    value = command[param]
                    if not (min_val <= value <= max_val):
                        return {
                            "valid": False,
                            "message": f"{param} must be between {min_val} and {max_val}",
                        }

        return {"valid": True}

    def _execute_component_command(self, command: Dict) -> Dict:
        """Route command to appropriate component handler"""
        component = command["component"]

        handlers = {
            "climate": self._execute_climate,
            "windows": self._execute_windows,
            "seats": self._execute_seats,
            "lighting": self._execute_lighting,
            "drive_mode": self._execute_drive_mode,
        }

        if component in handlers:
            return handlers[component](command)

        return {
            "success": False,
            "error": "UNKNOWN_COMPONENT",
            "message": f"Unknown component: {component}",
        }

    def _execute_climate(self, command: Dict) -> Dict:
        """Execute climate control command"""
        old_state = self.state["climate"].copy()
        action = command["action"]

        if action == "set_temperature":
            self.state["climate"]["temperature_set"] = command["value"]
            # Would start timer in production

        elif action == "set_fan_speed":
            self.state["climate"]["fan_speed"] = command["value"]

        elif action == "set_mode":
            self.state["climate"]["mode"] = command["value"]

        elif action == "toggle_ac":
            self.state["climate"]["ac_on"] = command["value"]

        elif action == "max_cool":
            self.state["climate"]["temperature_set"] = 60
            self.state["climate"]["fan_speed"] = 7
            self.state["climate"]["ac_on"] = True
            self.state["climate"]["recirc"] = True

        elif action == "max_heat":
            self.state["climate"]["temperature_set"] = 85
            self.state["climate"]["fan_speed"] = 7
            self.state["climate"]["heater_on"] = True

        elif action == "defrost":
            self.state["climate"]["defrost_front"] = True
            self.state["climate"]["defrost_rear"] = True
            self.state["climate"]["fan_speed"] = 5

        new_state = self.state["climate"].copy()

        # Calculate time to reach target
        temp_diff = abs(new_state["temperature_set"] - old_state["temperature_current"])
        estimated_time = temp_diff * 60 if temp_diff > 0 else 0

        return {
            "success": True,
            "message": self._format_climate_message(action, new_state),
            "state_change": {"from": old_state, "to": new_state},
            "estimated_time": estimated_time,
        }

    def _execute_windows(self, command: Dict) -> Dict:
        """Execute window control command"""
        old_state = self.state["windows"].copy()
        action = command["action"]

        if action == "set_position":
            target = command.get("target", "driver")
            position = command["value"]
            self.state["windows"][target] = position

        elif action == "vent_all":
            for window in ["driver", "passenger", "rear_left", "rear_right"]:
                self.state["windows"][window] = 20

        elif action == "close_all":
            for window in self.state["windows"]:
                if window != "child_lock":
                    self.state["windows"][window] = 0

        elif action == "express_down":
            target = command.get("target", "driver")
            self.state["windows"][target] = 100

        elif action == "express_up":
            target = command.get("target", "driver")
            self.state["windows"][target] = 0

        new_state = self.state["windows"].copy()

        # Calculate time (2 seconds for full travel)
        max_change = max(abs(new_state[w] - old_state[w]) for w in new_state if w != "child_lock")
        estimated_time = (max_change / 100) * 2 if max_change > 0 else 0

        return {
            "success": True,
            "message": self._format_window_message(action, new_state),
            "state_change": {"from": old_state, "to": new_state},
            "estimated_time": estimated_time,
        }

    def _execute_seats(self, command: Dict) -> Dict:
        """Execute seat control command"""
        old_state = self.state["seats"].copy()
        action = command["action"]
        seat = command.get("seat", "driver")

        if action == "adjust_position":
            if "forward" in command:
                self.state["seats"][seat]["position_forward"] = command["forward"]
            if "height" in command:
                self.state["seats"][seat]["position_height"] = command["height"]
            if "recline" in command:
                self.state["seats"][seat]["position_recline"] = command["recline"]

        elif action == "set_heating":
            self.state["seats"][seat]["heating"] = command["value"]

        elif action == "set_cooling":
            self.state["seats"][seat]["cooling"] = command["value"]

        elif action == "set_lumbar":
            self.state["seats"][seat]["lumbar"] = command["value"]

        elif action == "memory_save":
            # In production, would save to persistent storage
            position = command.get("position", 1)
            logger.info(f"Saved seat position {position} for {seat}")

        elif action == "memory_recall":
            # In production, would load from storage
            position = command.get("position", 1)
            logger.info(f"Recalled seat position {position} for {seat}")

        new_state = self.state["seats"].copy()

        return {
            "success": True,
            "message": self._format_seat_message(action, seat, new_state[seat]),
            "state_change": {"from": old_state, "to": new_state},
            "estimated_time": 3,  # Seat adjustment takes ~3 seconds
        }

    def _execute_lighting(self, command: Dict) -> Dict:
        """Execute lighting control command"""
        old_state = self.state["lighting"].copy()
        action = command["action"]

        if action == "set_headlights":
            self.state["lighting"]["headlights"] = command["value"]

        elif action == "toggle_fog":
            self.state["lighting"]["fog_lights"] = command.get(
                "value", not self.state["lighting"]["fog_lights"]
            )

        elif action == "set_interior":
            self.state["lighting"]["interior_dome"] = command["value"]

        elif action == "set_ambient":
            if "color" in command:
                self.state["lighting"]["ambient"]["color"] = command["color"]
            if "intensity" in command:
                self.state["lighting"]["ambient"]["intensity"] = command["intensity"]
            self.state["lighting"]["ambient"]["enabled"] = True

        elif action == "reading_light":
            target = command.get("target", "driver")
            self.state["lighting"]["reading"][target] = command.get("value", True)

        new_state = self.state["lighting"].copy()

        return {
            "success": True,
            "message": self._format_lighting_message(action, new_state),
            "state_change": {"from": old_state, "to": new_state},
            "estimated_time": 0,  # Lighting is instant
        }

    def _execute_drive_mode(self, command: Dict) -> Dict:
        """Execute drive mode command"""
        old_state = self.state["drive_mode"].copy()
        action = command["action"]

        if action == "set_mode":
            mode = command["value"]
            self.state["drive_mode"]["current"] = mode

            # Adjust related settings based on mode
            if mode == "sport":
                self.state["drive_mode"]["traction_control"] = True
                self.state["drive_mode"]["stability_control"] = True
            elif mode == "eco":
                pass  # Eco settings
            elif mode == "snow":
                self.state["drive_mode"]["traction_control"] = True
                self.state["drive_mode"]["stability_control"] = True

        elif action == "toggle_traction":
            self.state["drive_mode"]["traction_control"] = command.get(
                "value", not self.state["drive_mode"]["traction_control"]
            )

        elif action == "toggle_lane_assist":
            self.state["drive_mode"]["lane_assist"] = command.get(
                "value", not self.state["drive_mode"]["lane_assist"]
            )

        new_state = self.state["drive_mode"].copy()

        return {
            "success": True,
            "message": self._format_drive_mode_message(action, new_state),
            "state_change": {"from": old_state, "to": new_state},
            "estimated_time": 0,
        }

    def _generate_can_signals(self, command: Dict, result: Dict) -> List[str]:
        """Generate mock CAN bus signals"""
        signals = []
        component = command["component"]

        can_ids = {
            "climate": 0x260,
            "windows": 0x2A0,
            "seats": 0x3B0,
            "lighting": 0x3C0,
            "drive_mode": 0x4A0,
        }

        if component in can_ids and result["success"]:
            base_id = can_ids[component]
            timestamp = int(time.time() * 1000) % 0xFFFF

            # Command acknowledgment
            signals.append(f"{base_id:03X}#{timestamp:04X}01")

            # State change signal
            if "state_change" in result:
                value = command.get("value", 0)
                # Convert string values to numeric codes for CAN signal
                if isinstance(value, str):
                    # Map string values to numeric codes
                    value_map = {
                        "sport": 0x01,
                        "eco": 0x02,
                        "normal": 0x00,
                        "snow": 0x03,
                        "on": 0x01,
                        "off": 0x00,
                        True: 0x01,
                        False: 0x00,
                    }
                    value = value_map.get(value, 0x00)
                elif isinstance(value, bool):
                    value = 0x01 if value else 0x00
                signals.append(f"{base_id+1:03X}#{value:02X}00")

            # Status update
            signals.append(f"{base_id+2:03X}#0100")

        return signals

    def _format_climate_message(self, action: str, state: Dict) -> str:
        """Format user-friendly climate message"""
        if action == "set_temperature":
            return f"Climate temperature set to {state['temperature_set']}°F"
        elif action == "max_cool":
            return "Maximum cooling activated - temperature set to 60°F, fan at maximum"
        elif action == "max_heat":
            return "Maximum heating activated - temperature set to 85°F, fan at maximum"
        elif action == "defrost":
            return "Defrost mode activated for front and rear windows"
        else:
            return f"Climate adjusted: temp {state['temperature_set']}°F, fan speed {state['fan_speed']}"

    def _format_window_message(self, action: str, state: Dict) -> str:
        """Format user-friendly window message"""
        if action == "vent_all":
            return "All windows vented to 20% for fresh air"
        elif action == "close_all":
            return "All windows closed"
        elif action == "express_down":
            return "Window fully opened"
        elif action == "express_up":
            return "Window fully closed"
        else:
            positions = [f"{k}: {v}%" for k, v in state.items() if k != "child_lock" and v > 0]
            if positions:
                return f"Window positions: {', '.join(positions)}"
            return "Windows adjusted"

    def _format_seat_message(self, action: str, seat: str, state: Dict) -> str:
        """Format user-friendly seat message"""
        if action == "set_heating":
            level = state["heating"]
            return (
                f"{seat.capitalize()} seat heating set to level {level}"
                if level > 0
                else f"{seat.capitalize()} seat heating off"
            )
        elif action == "set_cooling":
            level = state["cooling"]
            return (
                f"{seat.capitalize()} seat cooling set to level {level}"
                if level > 0
                else f"{seat.capitalize()} seat cooling off"
            )
        elif action == "memory_save":
            return f"{seat.capitalize()} seat position saved"
        elif action == "memory_recall":
            return f"{seat.capitalize()} seat position recalled"
        else:
            return f"{seat.capitalize()} seat adjusted"

    def _format_lighting_message(self, action: str, state: Dict) -> str:
        """Format user-friendly lighting message"""
        if action == "set_headlights":
            return f"Headlights set to {state['headlights']}"
        elif action == "set_ambient":
            ambient = state["ambient"]
            return (
                f"Ambient lighting set to {ambient['color']} at {ambient['intensity']}% intensity"
            )
        elif action == "reading_light":
            return "Reading light adjusted"
        else:
            return "Lighting adjusted"

    def _format_drive_mode_message(self, action: str, state: Dict) -> str:
        """Format user-friendly drive mode message"""
        mode = state["current"]
        if action == "set_mode":
            messages = {
                "sport": "Sport mode engaged - enhanced throttle response and firmer steering",
                "eco": "Eco mode engaged - optimized for fuel efficiency",
                "normal": "Normal mode engaged - balanced performance and comfort",
                "snow": "Snow mode engaged - optimized traction for slippery conditions",
            }
            return messages.get(mode, f"Drive mode set to {mode}")
        else:
            return f"Drive settings adjusted"

    def _get_component_state(self, component: str) -> Dict:
        """Get current state of a component"""
        return self.state.get(component, {})

    def _add_to_history(self, command: Dict, result: Dict):
        """Add command and result to history"""
        self.state_history.append(
            {"timestamp": datetime.now().isoformat(), "command": command, "result": result}
        )

        # Keep only last 100 entries
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]

    def simulate_driving(self, speed: int, gear: str = "D"):
        """Simulate vehicle in motion for testing"""
        with self.state_lock:
            self.state["vehicle_info"]["speed"] = speed
            self.state["vehicle_info"]["gear"] = gear
            self.state["vehicle_info"]["parking_brake"] = False if speed > 0 else True
            self.state["vehicle_info"]["engine_on"] = True
            logger.info(f"Vehicle simulation: speed={speed}mph, gear={gear}")

    def get_display_state(self) -> str:
        """Get formatted state for display"""
        with self.state_lock:
            info = self.state["vehicle_info"]
            climate = self.state["climate"]
            windows = self.state["windows"]

            display = f"""
╔══════════════════════════════════════════════════════╗
║  {info['model']} - Virtual ECU State                  ║
╠══════════════════════════════════════════════════════╣
║ Speed: {info['speed']} mph | Gear: {info['gear']} | Engine: {'ON' if info['engine_on'] else 'OFF'}
║ Climate: {climate['temperature_set']}°F (current: {climate['temperature_current']}°F) | Fan: {climate['fan_speed']}
║ Windows: Driver: {windows['driver']}% | Passenger: {windows['passenger']}%
║ Drive Mode: {self.state['drive_mode']['current'].upper()}
╚══════════════════════════════════════════════════════╝"""
            return display


# Singleton instance
_ecu_instance = None
_ecu_lock = threading.Lock()


def get_virtual_ecu(vehicle_model: str = None) -> VirtualECU:
    """Get or create the global Virtual ECU instance"""
    global _ecu_instance

    with _ecu_lock:
        if _ecu_instance is None:
            # Determine vehicle model from environment or default
            import os

            if vehicle_model is None:
                driver_profile = os.environ.get("DRIVER_PROFILE", "guest").lower()
                if driver_profile == "sarah":
                    vehicle_model = "AutoDrive CX-7"
                elif driver_profile == "bob":
                    vehicle_model = "TechCar Model S"
                else:
                    vehicle_model = "TechCar Model X"

            _ecu_instance = VirtualECU(vehicle_model)

    return _ecu_instance
