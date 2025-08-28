# import strands to load tools
import json
import copy
import sys
import os
from typing import Dict, List, Any
import inspect
from jinja2 import Template

# Add the parent directory to the path so we can import the tools
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.cockpit import climate_control, window_control, seat_control, lighting_control, drive_mode

print("Loading tools...")
print(climate_control.tool_spec)
print(window_control.tool_spec)
print(seat_control.tool_spec)
print(lighting_control.tool_spec)
print(drive_mode.tool_spec)

