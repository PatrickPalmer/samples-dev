# Edge AI Agents - Architecture

## Directory Structure

The agents are organized by their primary function to improve modularity and maintainability:

```
agents/
├── cockpit/           # Direct vehicle control agents
│   ├── climate_control.py    # HVAC system control
│   ├── window_control.py     # Window and sunroof control
│   ├── seat_control.py       # Seat position and comfort
│   ├── lighting_control.py   # Interior/exterior lighting
│   └── drive_mode.py         # Driving dynamics modes
│
└── tools/            # Shared utilities
    └── model_selector.py     # Dynamic model routing
```

## Agent Categories

### 🎛️ Cockpit Control Agents
Direct vehicle control through natural language commands:
- **Climate Control**: Temperature, AC, defrost, fan speed
- **Window Control**: Individual/group window control with safety checks
- **Seat Control**: Position, heating/cooling, memory functions
- **Lighting Control**: Headlights, ambient, interior, reading lights
- **Drive Mode**: Sport/Eco/Normal/Snow modes, traction control

All cockpit agents interface with the Virtual ECU for state management and generate realistic CAN bus signals.

### 🔧 Tools
Shared utilities used across agents:
- **Model Selector**: Dynamic routing between local/cloud models based on query complexity

## Usage Example

```python
# Import cockpit control agents
from src.agents.cockpit import climate_control, window_control, seat_control
from src.agents.tools import select_model

# Use agents
response = climate_control("Set temperature to 72 degrees")
response = window_control("Open driver window")
response = seat_control("Move seat back")

# Model selection
model_choice = select_model("What's the weather forecast for next week?")
# Returns: {"provider": "bedrock", "reasoning": "Complex query requiring current data"}
```

## Data Dependencies

- **Vehicle Systems**: Virtual ECU state management in `src/data/vehicle_systems/`
- **Configuration**: Model settings in `src/config.py`

## Testing

Test files mirror the agent structure:
- `tests/test_cockpit_controls.py` - Cockpit agent tests
- `tests/test_tools.py` - Tool functionality tests