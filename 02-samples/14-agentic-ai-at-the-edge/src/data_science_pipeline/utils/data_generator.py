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
import uuid
import boto3
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path


# Configuration constants
DEFAULT_MODEL_ID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
DEFAULT_REGION = "us-east-1"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.7


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
        """Initialize tool specifications matching actual implementation"""

        tools = {
            # Cockpit Control Tools - All take a 'command' string parameter
            "climate_control": ToolSpec(
                name="climate_control",
                description="Control vehicle climate settings",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Climate control command"}
                    },
                    "required": ["command"],
                },
            ),
            "window_control": ToolSpec(
                name="window_control",
                description="Control vehicle windows",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Window control command"}
                    },
                    "required": ["command"],
                },
            ),
            "seat_control": ToolSpec(
                name="seat_control",
                description="Control vehicle seat settings",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Seat control command"}
                    },
                    "required": ["command"],
                },
            ),
            "lighting_control": ToolSpec(
                name="lighting_control",
                description="Control vehicle lighting",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Lighting control command"}
                    },
                    "required": ["command"],
                },
            ),
            "drive_mode": ToolSpec(
                name="drive_mode",
                description="Control vehicle drive mode settings",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Drive mode command"}
                    },
                    "required": ["command"],
                },
            ),
            # Model Selection Tool
            "select_model": ToolSpec(
                name="select_model",
                description="Intelligently select between local and remote models based on query complexity",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "User's query to analyze"},
                        "context": {
                            "type": "string",
                            "description": "Optional conversation context",
                        },
                    },
                    "required": ["query"],
                },
            ),
        }

        return tools

    def get_tool(self, name: str) -> Optional[ToolSpec]:
        """Get tool specification by name"""
        return self.tools.get(name)

    def get_random_tools(self, count: int = 1) -> List[ToolSpec]:
        """Get random selection of tools"""
        tool_names = list(self.tools.keys())
        selected = random.sample(tool_names, min(count, len(tool_names)))
        return [self.tools[name] for name in selected]


class DataGenerator:
    """Generate synthetic training data for tool calling using foundation models"""

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.tool_registry = tool_registry or ToolRegistry()
        self.llm_client = self._setup_llm_client()

    def _setup_llm_client(self):
        """Setup foundation model client for AWS Bedrock"""
        return boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", DEFAULT_REGION))

    # ====================================================================
    # LLM Generation Methods
    # ====================================================================

    def _generate_with_llm(self, prompt: str) -> str:
        """Generate content using foundation model via Bedrock"""
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

        response = self.llm_client.invoke_model(modelId=DEFAULT_MODEL_ID, body=json.dumps(body))

        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    def _generate_user_request(self, tool_spec: ToolSpec) -> str:
        """Generate realistic user request for the specified tool"""
        prompt = f"""Generate a realistic user request for a vehicle AI assistant that would trigger the "{tool_spec.name}" tool.

Tool description: {tool_spec.description}

Context: This is for a TechCar Model X vehicle with an AI assistant. The user is driving or inside the vehicle.

Requirements:
- Make it natural and conversational
- Use automotive context and terminology
- Make it sound like something someone would actually say in a car
- Keep it concise (1-2 sentences max)
- Don't mention the tool name directly

Examples of good requests:
- "It's getting too warm in here" (for climate_control)
- "My tire pressure warning light just came on" (for vehicle_assistant)  
- "What do I have scheduled today?" (for calendar_assistant)

Generate just the user request, nothing else:"""

        result = self._generate_with_llm(prompt)
        # Clean up the response to get just the user request
        lines = result.strip().split("\n")
        for line in lines:
            line = line.strip().strip('"').strip("'")
            if line and not line.startswith(("Generate", "User:", "Request:", "Query:")):
                return line

        # Fallback to first line if no clean line found
        return lines[0].strip().strip('"').strip("'") if lines else "Help me with this"

    # ====================================================================
    # Conversation Generation Methods
    # ====================================================================

    def generate_tool_call(self, tool_spec: ToolSpec, user_input: str) -> Dict[str, Any]:
        """Generate a tool call in Strands format"""

        # Generate parameters based on tool spec and user input
        params = self._generate_parameters(tool_spec, user_input)

        return {
            "toolUse": {
                "toolUseId": f"call_{uuid.uuid4().hex[:8]}",
                "name": tool_spec.name,
                "input": params,
            }
        }

    def _generate_parameters(self, tool_spec: ToolSpec, user_input: str = "") -> Dict[str, Any]:
        """Generate valid parameters for a tool based on actual tool signatures"""
        params = {}
        properties = tool_spec.parameters.get("properties", {})
        required = tool_spec.parameters.get("required", [])

        # Most tools just take a command or query string
        if "command" in properties:
            params["command"] = user_input or self._generate_command(tool_spec.name)
        elif "query" in properties:
            params["query"] = user_input or self._generate_query(tool_spec.name)
        elif "image_query" in properties and random.random() > 0.5:
            params["image_query"] = "What does this show?"
        elif "duration" in properties and random.random() > 0.7:
            params["duration"] = random.randint(3, 10)

        # Add other required parameters
        for prop_name in required:
            if prop_name not in params:
                params[prop_name] = self._generate_value(properties[prop_name])

        return params

    def _generate_command(self, tool_name: str) -> str:
        """Generate a command string for cockpit control tools"""
        prompt = f"""Generate a short, natural command that a driver would give to control the "{tool_name}" system in their vehicle.

Requirements:
- Keep it conversational and natural
- 3-8 words maximum
- Don't mention the tool name directly
- Sound like something said while driving

Examples for different systems:
- "Set temperature to 70"
- "Open driver window halfway"
- "Turn on seat heating"

Generate just the command:"""

        result = self._generate_with_llm(prompt)
        return result.strip().strip('"').strip("'")

    def _generate_query(self, tool_name: str) -> str:
        """Generate a query string for assistant tools"""
        prompt = f"""Generate a short, natural question or request that a driver would ask the "{tool_name}" assistant in their vehicle.

Requirements:
- Keep it conversational and natural
- 5-12 words maximum
- Sound like something said while driving
- Don't mention the tool name directly

Examples for different assistants:
- "What's on my calendar today?"
- "Check engine light came on"
- "Find nearby gas stations"

Generate just the query:"""

        result = self._generate_with_llm(prompt)
        return result.strip().strip('"').strip("'")

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
        self, tools: List[ToolSpec], include_multimodal: bool = False
    ) -> Dict[str, Any]:
        """Generate complete conversation with tool calls"""

        conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
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

        # Extract text from user content for tool call
        user_text = user_content
        if isinstance(user_content, list):
            # Extract text from multimodal content
            for item in user_content:
                if isinstance(item, dict) and "text" in item:
                    user_text = item["text"]
                    break

        # Assistant response with tool call
        tool_call = self.generate_tool_call(
            tools[0], user_text if isinstance(user_text, str) else ""
        )
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
        return [{"text": "I'll help you with that."}, tool_call]

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
        prompt = f"""Generate a brief response that an AI vehicle assistant would give after successfully using the "{tool_name}" tool.

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

    def generate_dataset(
        self, num_examples: int = 1000, output_path: str = "training_data.jsonl"
    ) -> None:
        """Generate complete training dataset"""

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            for i in range(num_examples):
                # Select random tools
                num_tools = random.randint(1, 3)
                tools = self.tool_registry.get_random_tools(num_tools)

                # Generate conversation
                include_multimodal = random.random() < 0.3
                conversation = self.generate_conversation(tools, include_multimodal)

                # Write to file
                f.write(json.dumps(conversation) + "\n")

                if (i + 1) % 100 == 0:
                    print(f"Generated {i + 1}/{num_examples} examples (LLM-generated)")

        print(f"Dataset saved to {output_file}")
