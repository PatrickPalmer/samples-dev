cockpit_tools = [
    {
        "name": "climate_control",
        "description": "Control vehicle climate system including temperature, fan speed, and air distribution",
        "inputSchema": {
            "json": {
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action: set_temperature, adjust_temperature, set_fan_speed, adjust_fan_speed, set_mode, toggle_ac, toggle_defrost, turn_off",
                    },
                    "temperature": {
                        "type": "integer",
                        "description": "Target temperature in Fahrenheit (60-85)",
                    },
                    "temperature_adjustment": {
                        "type": "integer",
                        "description": "Temperature change in degrees (-10 to +10)",
                    },
                    "fan_speed": {
                        "type": "integer",
                        "description": "Fan speed level (0-7)",
                    },
                    "fan_adjustment": {
                        "type": "integer",
                        "description": "Fan speed change (-3 to +3)",
                    },
                    "mode": {
                        "type": "string",
                        "description": "Climate mode: auto, heat, cool, defrost, vent",
                    },
                    "enable": {
                        "type": "boolean",
                        "description": "Enable/disable for toggle actions",
                    },
                },
                "required": ["action"],
                "type": "object",
            }
        },
    },
    {
        "name": "window_control",
        "description": "Control vehicle windows including driver, passenger, rear windows, and sunroof",
        "inputSchema": {
            "json": {
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action: open, close, set_position, vent, express_up, express_down, toggle_child_lock",
                    },
                    "target": {
                        "type": "string",
                        "description": "Target: driver, passenger, rear_left, rear_right, rear, all, sunroof",
                    },
                    "position": {
                        "type": "integer",
                        "description": "Window position percentage (0=closed, 100=fully open)",
                    },
                    "enable": {
                        "type": "boolean",
                        "description": "Enable/disable for toggle actions",
                    },
                },
                "required": ["action", "target"],
                "type": "object",
            }
        },
    },
    {
        "name": "seat_control",
        "description": "Control vehicle seat position, heating, cooling, and memory settings",
        "inputSchema": {
            "json": {
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action: adjust_position, set_heating, set_cooling, adjust_lumbar, save_memory, recall_memory",
                    },
                    "seat": {
                        "type": "string",
                        "description": "Seat: driver, passenger",
                    },
                    "position_type": {
                        "type": "string",
                        "description": "Position type: forward, height, tilt",
                    },
                    "adjustment": {
                        "type": "integer",
                        "description": "Position adjustment amount (-20 to +20)",
                    },
                    "heating_level": {
                        "type": "integer",
                        "description": "Seat heating level (0-3, 0=off)",
                    },
                    "cooling_level": {
                        "type": "integer",
                        "description": "Seat cooling level (0-3, 0=off)",
                    },
                    "lumbar_adjustment": {
                        "type": "integer",
                        "description": "Lumbar support adjustment (-5 to +5)",
                    },
                    "memory_slot": {
                        "type": "integer",
                        "description": "Memory slot number (1-3)",
                    },
                },
                "required": ["action", "seat"],
                "type": "object",
            }
        },
    },
    {
        "name": "lighting_control",
        "description": "Control vehicle interior and exterior lighting including headlights, ambient lighting, and reading lights",
        "inputSchema": {
            "json": {
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action: set_headlights, set_interior, set_ambient, set_reading, toggle_auto",
                    },
                    "headlight_mode": {
                        "type": "string",
                        "description": "Headlight mode: off, on, auto, high_beam",
                    },
                    "interior_brightness": {
                        "type": "integer",
                        "description": "Interior light brightness percentage (0-100)",
                    },
                    "ambient_color": {
                        "type": "string",
                        "description": "Ambient lighting color: white, blue, red, green, purple, orange",
                    },
                    "ambient_brightness": {
                        "type": "integer",
                        "description": "Ambient lighting brightness percentage (0-100)",
                    },
                    "reading_light": {
                        "type": "string",
                        "description": "Reading light: driver, passenger, rear_left, rear_right, all",
                    },
                    "enable": {
                        "type": "boolean",
                        "description": "Enable/disable for toggle actions",
                    },
                },
                "required": ["action"],
                "type": "object",
            }
        },
    },
    {
        "name": "drive_mode",
        "description": "Control vehicle drive mode and dynamics settings for different driving conditions",
        "inputSchema": {
            "json": {
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action: set_mode, toggle_traction, toggle_stability, toggle_lane_assist, toggle_cruise",
                    },
                    "mode": {
                        "type": "string",
                        "description": "Drive mode: normal, sport, eco, snow",
                    },
                    "enable": {
                        "type": "boolean",
                        "description": "Enable/disable for toggle actions",
                    },
                },
                "required": ["action"],
                "type": "object",
            }
        },
    },
]