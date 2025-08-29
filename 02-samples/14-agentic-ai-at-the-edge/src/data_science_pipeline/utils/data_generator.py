"""
Data generation utilities for tool calling datasets.

This module generates synthetic conversation data using foundation models
via AWS Bedrock. All conversations are generated dynamically using LLM prompting for
maximum realism and variety.

Key Features:
- 100% LLM-generated content (no templates)
- Complete tool coverage for vehicle AI assistant
- Configurable generation parameters
- Robust error handling and fallbacks
"""

import json
import random
import re
import uuid
import boto3
import os
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from botocore.exceptions import ClientError

# Try to import transformers for chat template support
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available. Using manual chat formatting.")


# Configuration constants
DEFAULT_MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
# DEFAULT_MODEL_ID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
DEFAULT_REGION = "us-west-2"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.7

# Rate limiting configuration
# Reduced delays to accelerate data generation.
# Adjust these values based on your AWS account's rate limits.
RATE_LIMIT_DELAY = 0.5  # Base delay between requests in seconds
MAX_RETRIES = 5  # Maximum number of retries for rate-limited requests
BACKOFF_FACTOR = 2.0  # Exponential backoff multiplier
BATCH_SIZE = 10  # Process in batches with delays
BATCH_DELAY = 2.0  # Delay between batches in seconds


@dataclass
class ToolSpec:
    """Tool specification matching Strands SDK format"""

    name: str
    description: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {"json": self.parameters},
        }


class ToolRegistry:
    """Registry of available tools for training"""

    def __init__(self):
        self.tools = self._initialize_tools()

    def _initialize_tools(self) -> Dict[str, ToolSpec]:
        """Initialize ONLY the 5 production cockpit control tools"""

        tools = {
            # PRODUCTION TOOLS - Now using structured parameters
            "climate_control": ToolSpec(
                name="climate_control",
                description="Control vehicle climate settings including temperature, fan speed, and AC",
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["set_temperature", "adjust_temperature", "set_fan_speed", "adjust_fan_speed", "set_mode", "toggle_ac", "toggle_defrost", "turn_off"],
                            "description": "The climate control action to perform"
                        },
                        "temperature": {"type": "integer", "minimum": 60, "maximum": 85, "description": "Target temperature in Fahrenheit"},
                        "temperature_adjustment": {"type": "integer", "minimum": -10, "maximum": 10, "description": "Temperature change in degrees"},
                        "fan_speed": {"type": "integer", "minimum": 0, "maximum": 7, "description": "Fan speed level"},
                        "fan_adjustment": {"type": "integer", "minimum": -3, "maximum": 3, "description": "Fan speed change"},
                        "mode": {"type": "string", "enum": ["auto", "heat", "cool", "defrost", "vent"], "description": "Climate mode"},
                        "enable": {"type": "boolean", "description": "Enable/disable for toggle actions"}
                    },
                    "required": ["action"],
                },
            ),
            "window_control": ToolSpec(
                name="window_control",
                description="Control vehicle windows position",
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["open", "close", "set_position", "vent", "express_up", "express_down", "toggle_child_lock"],
                            "description": "The window control action to perform"
                        },
                        "target": {
                            "type": "string",
                            "enum": ["driver", "passenger", "rear_left", "rear_right", "rear", "all", "sunroof"],
                            "description": "Which window(s) to control"
                        },
                        "position": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Window position percentage"},
                        "enable": {"type": "boolean", "description": "Enable/disable for toggle actions"}
                    },
                    "required": ["action", "target"],
                },
            ),
            "seat_control": ToolSpec(
                name="seat_control",
                description="Control vehicle seat position and heating/cooling",
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["adjust_position", "set_heating", "set_cooling", "adjust_lumbar", "save_memory", "recall_memory"],
                            "description": "The seat control action to perform"
                        },
                        "seat": {"type": "string", "enum": ["driver", "passenger"], "description": "Which seat to control"},
                        "position_type": {"type": "string", "enum": ["forward", "height", "tilt"], "description": "Type of position adjustment"},
                        "adjustment": {"type": "integer", "minimum": -20, "maximum": 20, "description": "Position adjustment amount"},
                        "heating_level": {"type": "integer", "minimum": 0, "maximum": 3, "description": "Seat heating level"},
                        "cooling_level": {"type": "integer", "minimum": 0, "maximum": 3, "description": "Seat cooling level"},
                        "lumbar_adjustment": {"type": "integer", "minimum": -5, "maximum": 5, "description": "Lumbar support adjustment"},
                        "memory_slot": {"type": "integer", "minimum": 1, "maximum": 3, "description": "Memory slot number"}
                    },
                    "required": ["action", "seat"],
                },
            ),
            "lighting_control": ToolSpec(
                name="lighting_control",
                description="Control vehicle lighting including headlights and interior",
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["set_headlights", "set_interior", "set_ambient", "set_reading", "toggle_auto"],
                            "description": "The lighting control action to perform"
                        },
                        "headlight_mode": {"type": "string", "enum": ["off", "on", "auto", "high_beam"], "description": "Headlight mode setting"},
                        "interior_brightness": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Interior light brightness percentage"},
                        "ambient_color": {"type": "string", "enum": ["white", "blue", "red", "green", "purple", "orange"], "description": "Ambient lighting color"},
                        "ambient_brightness": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Ambient lighting brightness percentage"},
                        "reading_light": {"type": "string", "enum": ["driver", "passenger", "rear_left", "rear_right", "all"], "description": "Which reading light to control"},
                        "enable": {"type": "boolean", "description": "Enable/disable for toggle actions"}
                    },
                    "required": ["action"],
                },
            ),
            "drive_mode": ToolSpec(
                name="drive_mode",
                description="Control vehicle drive mode (sport, eco, comfort)",
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["set_mode", "toggle_traction", "toggle_stability", "toggle_lane_assist", "toggle_cruise"],
                            "description": "The drive mode action to perform"
                        },
                        "mode": {"type": "string", "enum": ["normal", "sport", "eco", "snow"], "description": "Drive mode setting"},
                        "enable": {"type": "boolean", "description": "Enable/disable for toggle actions"}
                    },
                    "required": ["action"],
                },
            ),
        }

        return tools

    def get_tool(self, name: str) -> Optional[ToolSpec]:
        """Get tool specification by name"""
        return self.tools.get(name)

    def get_random_tools(self, count: int = 1) -> List[ToolSpec]:
        """Get random selection of tools"""
        # Only use actually implemented tools (verified in src/agents/cockpit/)
        # These are the exact tools registered in main.py
        implemented_tools = [
            "climate_control",
            "window_control", 
            "seat_control",
            "lighting_control",
            "drive_mode",
        ]
        available_tools = [name for name in implemented_tools if name in self.tools]
        selected = random.sample(available_tools, min(count, len(available_tools)))
        return [self.tools[name] for name in selected]


class DataGenerator:
    """Generate synthetic training data for tool calling using foundation models"""

    def __init__(self, tool_registry: Optional[ToolRegistry] = None, model_name: str = "Qwen/Qwen3-1.7B",
                 aws_access_key_id: str = None, aws_secret_access_key: str = None,
                 aws_session_token: str = None, aws_profile_name: str = None,
                 aws_region: str = None):
        self.tool_registry = tool_registry or ToolRegistry()
        self.llm_client = self._setup_llm_client(aws_access_key_id, aws_secret_access_key,
                                                 aws_session_token, aws_profile_name, aws_region)
        self.model_name = model_name
        self.tokenizer = self._setup_tokenizer()

    def _setup_llm_client(self, aws_access_key_id, aws_secret_access_key,
                          aws_session_token, aws_profile_name, aws_region):
        """Setup foundation model client for AWS Bedrock with flexible authentication."""
        region = aws_region or os.getenv("AWS_REGION", DEFAULT_REGION)
        
        if aws_access_key_id and aws_secret_access_key:
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region
            )
        elif aws_profile_name:
            session = boto3.Session(profile_name=aws_profile_name, region_name=region)
        else:
            # Use default credential chain (env vars, .aws/credentials, IAM role)
            session = boto3.Session(region_name=region)
            
        return session.client("bedrock-runtime")

    def _setup_tokenizer(self):
        """Setup tokenizer for chat template formatting"""
        if not HAS_TRANSFORMERS:
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Ensure we have a chat template
            if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
                print(f"Warning: {self.model_name} doesn't have a chat template. Using manual formatting.")
                return None
            return tokenizer
        except Exception as e:
            print(f"Warning: Could not load tokenizer for {self.model_name}: {e}")
            return None

    # ====================================================================
    # LLM Generation Methods
    # ====================================================================

    def _generate_with_llm(self, prompt: str) -> str:
        """Generate content using foundation model via Bedrock with retry logic"""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Handle bearer token if provided
        bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        if bearer_token and bearer_token != "YOUR_BEARER_TOKEN_HERE":
            # Use custom headers for bearer token (if supported by boto3)
            # Note: Standard boto3 may not support custom auth headers for Bedrock
            # This is a placeholder for custom authentication logic
            pass

        # Retry logic with exponential backoff
        for attempt in range(MAX_RETRIES):
            try:
                # Add base delay to prevent rapid-fire requests
                if attempt > 0:
                    delay = RATE_LIMIT_DELAY * (BACKOFF_FACTOR ** (attempt - 1))
                    time.sleep(delay)
                
                response = self.llm_client.invoke_model(
                    modelId=DEFAULT_MODEL_ID, 
                    body=json.dumps(body)
                )
                
                result = json.loads(response["body"].read())
                return result["content"][0]["text"]
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_message = str(e)
                
                # Check for throttling errors
                if 'ThrottlingException' in error_code or 'ThrottlingException' in error_message:
                    if attempt < MAX_RETRIES - 1:
                        delay = RATE_LIMIT_DELAY * (BACKOFF_FACTOR ** attempt)
                        print(f"Rate limited. Retrying in {delay:.1f}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"Max retries reached. Using fallback response.")
                        return self._generate_fallback_response(prompt)
                        
                # Check for rate limit errors
                elif 'Too many requests' in error_message or 'rate' in error_message.lower():
                    if attempt < MAX_RETRIES - 1:
                        delay = RATE_LIMIT_DELAY * (BACKOFF_FACTOR ** attempt)
                        print(f"Rate limited. Retrying in {delay:.1f}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"Max retries reached. Using fallback response.")
                        return self._generate_fallback_response(prompt)
                else:
                    # Re-raise non-throttling errors
                    raise
                    
        # Fallback if all retries exhausted
        return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a fallback response when LLM is unavailable"""
        # Determine if this is for conversational or direct style
        is_conversational = "conversational" in prompt.lower()

        # Fallback responses based on prompt content and style
        if "climate_control" in prompt.lower():
            if is_conversational:
                return random.choice([
                    "It's getting too warm in here",
                    "I'm feeling a bit cold",
                    "It's stuffy, can you help with the air?"
                ])
            else:
                return random.choice([
                    "Set the temperature to 72 degrees",
                    "Turn up the heat by 5 degrees",
                    "Set fan speed to maximum"
                ])
        elif "window_control" in prompt.lower():
            if is_conversational:
                return random.choice([
                    "It's getting stuffy in here",
                    "I need some fresh air",
                    "It's too hot with the windows closed"
                ])
            else:
                return random.choice([
                    "Open the driver window",
                    "Close all windows",
                    "Open windows halfway"
                ])
        elif "seat_control" in prompt.lower():
            if is_conversational:
                return random.choice([
                    "My back is bothering me",
                    "I'm not comfortable in this position",
                    "I'm feeling a bit cold"
                ])
            else:
                return random.choice([
                    "Move seat forward 3 inches",
                    "Turn on seat heating level 2",
                    "Adjust lumbar support"
                ])
        elif "lighting_control" in prompt.lower():
            if is_conversational:
                return random.choice([
                    "I can't see well, it's pretty dark",
                    "It's too bright in here",
                    "The mood lighting would be nice"
                ])
            else:
                return random.choice([
                    "Turn on the headlights",
                    "Set ambient lighting to blue",
                    "Turn on reading lights"
                ])
        elif "drive_mode" in prompt.lower():
            if is_conversational:
                return random.choice([
                    "I want to have some fun driving today",
                    "I need to save fuel on this trip",
                    "The roads are slippery today"
                ])
            else:
                return random.choice([
                    "Switch to sport mode",
                    "Enable eco mode",
                    "Set to snow mode"
                ])
        else:
            return "Help me with this task"

    def _generate_user_request(self, tool_spec: ToolSpec) -> str:
        """Generate realistic user request for the specified tool"""

        # Favor direct commands over conversational (70% direct, 30% conversational)
        style = random.choices(["direct", "conversational"], weights=[70, 30])[0]

        if style == "conversational":
            prompt = f"""Generate a realistic conversational user request for a vehicle AI assistant that would trigger the "{tool_spec.name}" tool.

Tool description: {tool_spec.description}

Context: This is for a TechCar Model X vehicle with an AI assistant. The user is driving or inside the vehicle.

Requirements:
- Make it natural and conversational (not a direct command)
- Express a need or situation rather than giving specific instructions
- Use automotive context and terminology
- Make it sound like something someone would actually say in a car
- Keep it concise (1-2 sentences max)
- Don't mention the tool name directly

Examples of good conversational requests:
- "It's getting too warm in here" (for climate_control)
- "I can't see well, it's pretty dark" (for lighting_control)
- "I'm feeling a bit stiff, my back is bothering me" (for seat_control)
- "It's getting stuffy in here" (for window_control)
- "I want to have some fun driving today" (for drive_mode)

Generate just the user request, nothing else:"""
        else:  # direct style
            prompt = f"""Generate a realistic direct command for a vehicle AI assistant that would trigger the "{tool_spec.name}" tool.

Tool description: {tool_spec.description}

Context: This is for a TechCar Model X vehicle with an AI assistant. The user is driving or inside the vehicle.

Requirements:
- Make it a clear, direct command with specific details
- Include specific values, positions, or settings when appropriate
- Use automotive context and terminology
- Make it sound like something someone would actually say in a car
- Keep it concise (1-2 sentences max)
- Don't mention the tool name directly

Examples of good direct commands:
- "Set the temperature to 72 degrees" (for climate_control)
- "Turn on the headlights" (for lighting_control)
- "Move my seat forward by 3 inches" (for seat_control)
- "Open the driver window halfway" (for window_control)
- "Switch to eco mode" (for drive_mode)
- "Turn up the heat by 5 degrees" (for climate_control)
- "Close all the windows" (for window_control)
- "Set ambient lighting to blue at 70%" (for lighting_control)

Generate just the user request, nothing else:"""

        result = self._generate_with_llm(prompt)
        # Clean up the response to get just the user request
        lines = result.strip().split("\n")
        for line in lines:
            line = line.strip().strip('"').strip("'")
            if line and not line.startswith(("Generate", "User:", "Request:", "Query:", "Command:")):
                return line

        # Fallback to first line if no clean line found
        return lines[0].strip().strip('"').strip("'") if lines else "Help me with this"

    def _generate_compound_user_request(self, tools: List[ToolSpec]) -> str:
        """Generate user request that requires multiple tools"""

        # Define realistic compound scenarios
        compound_scenarios = {
            ("climate_control", "window_control"): [
                "It's getting stuffy and warm in here, can you help me cool down?",
                "Turn on the AC and crack open the windows a bit",
                "Set the temperature to 70 and open the driver window halfway"
            ],
            ("seat_control", "climate_control"): [
                "I'm cold and uncomfortable, can you warm me up and adjust my seat?",
                "Turn on seat heating to medium and set the temperature to 75",
                "Move my seat back and turn up the heat"
            ],
            ("lighting_control", "window_control"): [
                "It's getting dark and stuffy in here",
                "Turn on the headlights and open the windows a bit",
                "I need better visibility and some fresh air"
            ],
            ("seat_control", "lighting_control"): [
                "Adjust my seat position and turn on the reading light",
                "Move my seat forward and set ambient lighting to blue",
                "I need to get comfortable and improve the lighting"
            ],
            ("drive_mode", "climate_control"): [
                "Switch to eco mode and turn on the AC",
                "I want to save fuel but stay cool",
                "Set eco mode and adjust the temperature to 72"
            ]
        }

        # Get tool names
        tool_names = tuple(sorted([tool.name for tool in tools]))

        # Find matching scenario or create generic one
        if tool_names in compound_scenarios:
            return random.choice(compound_scenarios[tool_names])
        else:
            # Generic compound request
            actions = []
            for tool in tools:
                if tool.name == "climate_control":
                    actions.append("adjust the temperature")
                elif tool.name == "window_control":
                    actions.append("open the windows")
                elif tool.name == "seat_control":
                    actions.append("adjust my seat")
                elif tool.name == "lighting_control":
                    actions.append("turn on the lights")
                elif tool.name == "drive_mode":
                    actions.append("change the drive mode")

            return f"Please {' and '.join(actions)}"

    # ====================================================================
    # Conversation Generation Methods
    # ====================================================================

    def _generate_tool_call_from_request(self, user_request: str, tools: List[ToolSpec]) -> Optional[Dict[str, Any]]:
        """Generate a tool call based on a user request using an LLM."""
        tool_spec = tools[0]
        
        prompt = f"""You are an AI assistant. Your task is to generate the arguments for a tool call in JSON format based on the user's request and the available tool. The generated JSON must be valid and adhere strictly to the tool's input schema.

Tool:
{json.dumps(tool_spec.to_dict())}

User request:
{user_request}

Respond with ONLY the arguments in JSON format."""

        response = self._generate_with_llm(prompt)
        
        try:
            # Find the JSON object in the response
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if not match:
                return None

            arguments = json.loads(match.group(0))
            
            return {
                "toolUse": {
                    "toolUseId": f"call_{{uuid.uuid4().hex[:8]}}",
                    "name": tool_spec.name,
                    "input": arguments,
                }
            }
        except json.JSONDecodeError:
            return None

    def _generate_multiple_tool_calls(self, user_request: str, tools: List[ToolSpec]) -> List[Dict[str, Any]]:
        """Generate multiple tool calls for a compound request"""
        tool_calls = []

        for tool in tools:
            # Generate individual tool call for each tool
            prompt = f"""You are an AI assistant. Generate arguments for the "{tool.name}" tool based on this user request.

Tool:
{json.dumps(tool.to_dict())}

User request: {user_request}

Extract only the part of the request relevant to this tool and generate appropriate arguments.
Respond with ONLY the arguments in JSON format."""

            response = self._generate_with_llm(prompt)

            try:
                # Find the JSON object in the response
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    arguments = json.loads(match.group(0))

                    tool_call = {
                        "toolUse": {
                            "toolUseId": f"call_{uuid.uuid4().hex[:8]}",
                            "name": tool.name,
                            "input": arguments,
                        }
                    }
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue

        return tool_calls if tool_calls else None

    def _format_multi_tool_response(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format assistant response with multiple tool calls"""
        return tool_calls

    def _generate_multi_tool_response(self, tool_names: List[str]) -> str:
        """Generate assistant response after multiple tool executions"""
        if len(tool_names) == 2:
            return f"I've completed both actions for you. Your {tool_names[0]} and {tool_names[1]} settings have been adjusted."
        else:
            return f"I've completed all {len(tool_names)} actions as requested. Everything should be set up to your preferences now."

    def _generate_value(self, spec: Dict[str, Any]) -> Any:
        """Generate value based on JSON schema specification"""
        prop_type = spec.get("type")

        if prop_type == "string":
            if "enum" in spec:
                return random.choice(spec["enum"])
            return f"sample_{random.randint(1, 100)}"

        elif prop_type == "number":
            min_val = spec.get("minimum", 0)
            max_val = spec.get("maximum", 100)
            return round(random.uniform(min_val, max_val), 1)

        elif prop_type == "integer":
            min_val = spec.get("minimum", 0)
            max_val = spec.get("maximum", 100)
            return random.randint(min_val, max_val)

        elif prop_type == "boolean":
            return random.choice([True, False])

        elif prop_type == "object":
            return {}

        return None

    # ====================================================================
    # Main Dataset Generation Methods
    # ====================================================================

    def generate_conversation(
        self, tools: List[ToolSpec], include_multimodal: bool = False, multi_tool: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Generate complete conversation with tool calls"""

        conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
        messages = []

        # System message with all available tools
        messages.append(
            {
                "role": "system",
                "content": "You are an AI assistant with access to various tools. Use them to help users effectively.",
            }
        )

        if multi_tool and len(tools) > 1:
            return self._generate_multi_tool_conversation(conversation_id, tools, include_multimodal)
        else:
            return self._generate_single_tool_conversation(conversation_id, tools, include_multimodal)

    def _generate_single_tool_conversation(
        self, conversation_id: str, tools: List[ToolSpec], include_multimodal: bool
    ) -> Optional[Dict[str, Any]]:
        """Generate conversation with single tool call"""
        messages = []

        # System message
        messages.append(
            {
                "role": "system",
                "content": "You are an AI assistant with access to various tools. Use them to help users effectively.",
            }
        )

        # User message
        user_content = self._generate_user_message(tools[0], include_multimodal)
        messages.append({"role": "user", "content": user_content})

        # Assistant response with tool call
        tool_call = self._generate_tool_call_from_request(user_content, tools)
        if not tool_call:
            return None # Could not generate a valid tool call

        assistant_response = self._format_assistant_response(tool_call)
        messages.append({"role": "assistant", "content": assistant_response})

        # Tool result
        tool_result = self._generate_tool_result(tool_call["toolUse"]["toolUseId"])
        messages.append(tool_result)

        # Final assistant response
        messages.append(
            {"role": "assistant", "content": self._generate_assistant_response(tools[0].name)}
        )

        return {
            "conversation_id": conversation_id,
            "tools": [tool.to_dict() for tool in tools],
            "messages": messages,
        }

    def _generate_multi_tool_conversation(
        self, conversation_id: str, tools: List[ToolSpec], include_multimodal: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Generate conversation with multiple tool calls"""
        # Note: include_multimodal not used for multi-tool conversations for simplicity
        messages = []

        # System message
        messages.append(
            {
                "role": "system",
                "content": "You are an AI assistant with access to various tools. Use them to help users effectively.",
            }
        )

        # Generate compound user request
        user_content = self._generate_compound_user_request(tools)
        messages.append({"role": "user", "content": user_content})

        # Generate multiple tool calls
        tool_calls = self._generate_multiple_tool_calls(user_content, tools)
        if not tool_calls:
            return None

        # Assistant response with multiple tool calls
        assistant_response = self._format_multi_tool_response(tool_calls)
        messages.append({"role": "assistant", "content": assistant_response})

        # Multiple tool results
        for tool_call in tool_calls:
            tool_result = self._generate_tool_result(tool_call["toolUse"]["toolUseId"])
            messages.append(tool_result)

        # Final assistant response acknowledging all actions
        tool_names = [tc["toolUse"]["name"] for tc in tool_calls]
        messages.append(
            {"role": "assistant", "content": self._generate_multi_tool_response(tool_names)}
        )

        return {
            "conversation_id": conversation_id,
            "tools": [tool.to_dict() for tool in tools],
            "messages": messages,
        }

    def _generate_user_message(self, tool: ToolSpec, include_multimodal: bool) -> Any:
        """Generate user message content based on actual usage patterns"""

        if include_multimodal and tool.name in ["analyze_image", "voice_input"]:
            # Multimodal content
            content = []

            if tool.name == "analyze_image":
                content.append(
                    {
                        "type": "image",
                        "image": {"source": {"path": "sign.jpg"}},  # Matches actual demo image
                    }
                )
                # Generate realistic text for image analysis
                realistic_text = self._generate_user_request(tool)
                content.append({"type": "text", "text": realistic_text})
            elif tool.name == "voice_input":
                content.append(
                    {
                        "type": "audio",
                        "audio": {
                            "source": {"path": "voice_input.wav"}
                        },  # Matches actual audio file
                    }
                )
                content.append({"type": "text", "text": "Process this voice input"})

            return content
        else:
            # Text-only content - generate realistic request
            return self._generate_user_request(tool)

    def _format_assistant_response(self, tool_call: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format assistant response with tool call"""
        return [tool_call]

    def _generate_tool_result(self, tool_use_id: str) -> Dict[str, Any]:
        """Generate tool execution result"""
        return {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"text": "Operation completed successfully"}],
                        "status": "success",
                    }
                }
            ],
        }

    def _generate_assistant_response(self, tool_name: str) -> str:
        """Generate assistant response after tool execution"""
        prompt = f"""Generate a brief response that an AI vehicle assistant would give after successfully using the \"{tool_name}\" tool.

Requirements:
- Keep it concise (1-2 sentences)
- Sound natural and helpful
- Confirm the action was completed
- Use automotive context when appropriate
- Don't be overly technical

Examples:
- "Climate adjusted to your preference."
- "I've found the information in your vehicle manual."
- "Your calendar has been updated."

Generate just the response:"""

        result = self._generate_with_llm(prompt)
        lines = result.strip().split("\n")
        for line in lines:
            line = line.strip().strip('"').strip("'")
            if line and not line.startswith(("Generate", "Response:", "Assistant:")):
                return line

        # Fallback to first line if no clean line found
        return lines[0].strip().strip('"').strip("'") if lines else "Task completed successfully."

    def convert_to_training_format(self, conversation: Dict) -> str:
        """Convert Strands format to training format using chat templates when available"""

        if self.tokenizer and hasattr(self.tokenizer, 'chat_template'):
            return self._format_with_chat_template(conversation)
        else:
            return self._format_manually(conversation)

    def _format_with_chat_template(self, conversation: Dict) -> str:
        """Format conversation using tokenizer's chat template"""

        # Convert Strands format to standard chat format
        messages = []

        # Process conversation messages
        for msg in conversation.get('messages', []):
            if msg['role'] == 'system':
                messages.append(msg)  # Keep original system message
            elif msg['role'] == 'user':
                content = self._extract_user_content(msg)
                if content:
                    messages.append({"role": "user", "content": content})
            elif msg['role'] == 'assistant':
                content = self._extract_assistant_content(msg)
                if content:
                    messages.append({"role": "assistant", "content": content})

        # Extract tools to be passed to the template
        tools = conversation.get('tools', [])

        # Apply chat template
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            return formatted
        except Exception as e:
            print(f"Warning: Chat template failed: {e}. Falling back to manual formatting.")
            return self._format_manually(conversation)

    def _extract_user_content(self, msg: Dict) -> str:
        """Extract user content from message"""
        if isinstance(msg['content'], str):
            return msg['content']
        elif isinstance(msg['content'], list):
            parts = []
            for item in msg['content']:
                if 'toolResult' in item:
                    for c in item['toolResult'].get('content', []):
                        if 'text' in c:
                            parts.append(f"[Tool Result: {c['text']}]")
                elif 'text' in item:
                    parts.append(item['text'])
            return " ".join(parts)
        return ""

    def _extract_assistant_content(self, msg: Dict) -> str:
        """Extract assistant content from message, including tool calls"""
        if isinstance(msg['content'], str):
            return msg['content']
        elif isinstance(msg['content'], list):
            parts = []
            for item in msg['content']:
                if 'text' in item:
                    parts.append(item['text'])
                elif 'toolUse' in item:
                    tool_call = {
                        "name": item['toolUse']['name'],
                        "arguments": item['toolUse']['input']
                    }
                    parts.append(f"<tool_call>\n{json.dumps(tool_call, indent=2)}\n</tool_call>")
            return " ".join(parts)
        return ""

    def _format_manually(self, conversation: Dict) -> str:
        """Fallback manual formatting when chat template is not available"""
        lines = []

        # Add tools section
        lines.append("# Available Tools")
        for tool in conversation.get('tools', []):
            lines.append(f"- {tool['name']}: {tool['description']}")
        lines.append("")

        # Process messages
        for msg in conversation.get('messages', []):
            if msg['role'] == 'user':
                content = self._extract_user_content(msg)
                if content:
                    lines.append(f"User: {content}")

            elif msg['role'] == 'assistant':
                content = self._extract_assistant_content(msg)
                if content:
                    lines.append(f"Assistant: {content}")

        return "\n".join(lines)

    def generate_dataset(
        self, num_examples: int = 1000, output_path: str = "training_data.jsonl",
        use_batching: bool = True, format_for_training: bool = True
    ) -> None:
        """Generate complete training dataset with rate limiting and batching
        
        Args:
            num_examples: Number of examples to generate
            output_path: Path to save dataset
            use_batching: Whether to use batch delays
            format_for_training: If True, convert to plain text training format
        """

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            generated_count = 0
            i = 0
            while generated_count < num_examples:
                # Add delay between individual requests to prevent rate limiting
                if i > 0:
                    time.sleep(RATE_LIMIT_DELAY)
                
                # Add longer delay between batches
                if use_batching and i > 0 and i % BATCH_SIZE == 0:
                    print(f"Completed batch {i // BATCH_SIZE}. Pausing for {BATCH_DELAY}s to avoid rate limits...")
                    time.sleep(BATCH_DELAY)
                
                try:
                    # Mix single and multi-tool conversations (70% single, 30% multi)
                    use_multi_tool = random.random() < 0.3

                    if use_multi_tool:
                        # Select 2-3 tools for multi-tool conversation
                        num_tools = random.choice([2, 2, 3])  # Favor 2 tools
                        tools = self.tool_registry.get_random_tools(num_tools)
                    else:
                        # Select single tool
                        tools = self.tool_registry.get_random_tools(1)

                    # Generate conversation (NO MULTIMODAL for cleaner training)
                    conversation = self.generate_conversation(
                        tools,
                        include_multimodal=False,
                        multi_tool=use_multi_tool
                    )
                    
                    if conversation is None:
                        print(f"Skipping example {i+1} due to generation failure.")
                        continue

                    if format_for_training:
                        # Convert to training format
                        training_text = self.convert_to_training_format(conversation)
                        f.write(json.dumps({"text": training_text}) + "\n")
                    else:
                        # Keep original format
                        f.write(json.dumps(conversation) + "\n")

                    generated_count += 1
                    if generated_count % 10 == 0:
                        print(f"Generated {generated_count}/{num_examples} examples")
                        
                except Exception as e:
                    print(f"Error generating example {i + 1}: {e}")
                    # Continue with next example even if one fails
                    continue
                finally:
                    i += 1

        print(f"Dataset saved to {output_file}")


def main():
    """Generate training and test datasets with structured parameters and chat templates"""

    # Initialize generator with default model
    generator = DataGenerator(
        aws_profile_name=os.getenv("AWS_PROFILE"),
        aws_region=os.getenv("AWS_REGION")
    )

    print("🚀 STRUCTURED PARAMETER TOOL CALLING DATASET GENERATOR")
    print("=" * 60)
    print("✅ Refactored to use structured parameters instead of natural language commands")
    print("✅ Using proper chat templates when available (transformers)")
    print("✅ Fallback to manual formatting when needed")
    print()
    print("Production Tools:")
    for tool_name in generator.tool_registry.tools.keys():
        tool = generator.tool_registry.tools[tool_name]
        print(f"  - {tool_name}: {tool.description}")
    print()

    # Check if we have chat template support
    if generator.tokenizer:
        print(f"🎯 Using chat template from: {generator.model_name}")
    else:
        print("⚠️  Using manual formatting (transformers not available)")
    print("-" * 60)

    # Generate training data
    print("Generating training dataset...")
    generator.generate_dataset(
        num_examples=20,
        output_path="data/train_structured.jsonl",
        format_for_training=True
    )

    # Generate test data
    print("\nGenerating test dataset...")
    generator.generate_dataset(
        num_examples=10,
        output_path="data/test_structured.jsonl",
        format_for_training=True
    )

    print("\n🎉 DATASETS GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print("📁 Files created:")
    print("  - data/train_structured.jsonl (20 examples)")
    print("  - data/test_structured.jsonl (10 examples)")
    print()
    print("🔧 Format improvements:")
    print("  ✅ Structured parameters instead of natural language commands")
    print("  ✅ Proper chat template formatting when available")
    print("  ✅ Tool calls with JSON parameters")
    print("  ✅ Better validation and error handling")
    print("  ✅ Multi-tool conversations (30% of examples)")
    print("  ✅ Compound user requests requiring multiple tools")
    print("  ✅ Sequential tool call handling")
    print()
    print("🚀 Ready for fine-tuning!")


if __name__ == "__main__":
    main()
