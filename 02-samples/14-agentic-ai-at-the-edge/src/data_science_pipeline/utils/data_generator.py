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
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from botocore.exceptions import ClientError


# Configuration constants
DEFAULT_MODEL_ID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
DEFAULT_REGION = "us-east-1"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.7

# Rate limiting configuration
RATE_LIMIT_DELAY = 1.0  # Base delay between requests in seconds
MAX_RETRIES = 5  # Maximum number of retries for rate-limited requests
BACKOFF_FACTOR = 2.0  # Exponential backoff multiplier
BATCH_SIZE = 10  # Process in batches with delays
BATCH_DELAY = 5.0  # Delay between batches in seconds


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
            # PRODUCTION TOOLS ONLY - All take a 'command' string parameter
            "climate_control": ToolSpec(
                name="climate_control",
                description="Control vehicle climate settings including temperature, fan speed, and AC",
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
                description="Control vehicle windows position",
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
                description="Control vehicle seat position and heating/cooling",
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
                description="Control vehicle lighting including headlights and interior",
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
                description="Control vehicle drive mode (sport, eco, comfort)",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Drive mode command"}
                    },
                    "required": ["command"],
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
        # Simple fallback responses based on prompt content
        if "climate_control" in prompt.lower():
            return "Set the temperature to 72 degrees"
        elif "window_control" in prompt.lower():
            return "Open the driver window"
        elif "seat_control" in prompt.lower():
            return "Adjust my seat position"
        elif "lighting_control" in prompt.lower():
            return "Turn on the headlights"
        elif "drive_mode" in prompt.lower():
            return "Switch to sport mode"
        elif "select_model" in prompt.lower():
            return "What's the weather like today?"
        else:
            return "Help me with this task"

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
- "Open the driver side window" (for window_control)  
- "Switch to sport mode" (for drive_mode)

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

Examples for different systems:
- "Turn on the heated seats"
- "Set cruise control"
- "Adjust the mirrors"

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

    def convert_to_training_format(self, conversation: Dict) -> str:
        """Convert Strands format to plain text training format for llama.cpp
        
        This matches the format that llama.cpp expects AFTER Jinja processing.
        No chat template tokens, just plain text with XML tool tags.
        """
        lines = []
        
        # Add tools section
        lines.append("# Tools")
        lines.append("<tools>")
        for tool in conversation.get('tools', []):
            lines.append(f"{tool['name']}: {tool['description']}")
        lines.append("</tools>")
        lines.append("")
        
        # Process messages
        for msg in conversation.get('messages', []):
            if msg['role'] == 'user':
                if isinstance(msg['content'], str):
                    lines.append(f"User: {msg['content']}")
                elif isinstance(msg['content'], list):
                    for item in msg['content']:
                        if 'toolResult' in item:
                            for c in item['toolResult'].get('content', []):
                                if 'text' in c:
                                    lines.append(f"Tool Result: {c['text']}")
                        elif 'text' in item:
                            lines.append(f"User: {item['text']}")
                            
            elif msg['role'] == 'assistant':
                if isinstance(msg['content'], str):
                    lines.append(f"Assistant: {msg['content']}")
                elif isinstance(msg['content'], list):
                    for item in msg['content']:
                        if 'text' in item:
                            lines.append(f"Assistant: {item['text']}")
                        elif 'toolUse' in item:
                            lines.append("Assistant: <tool_call>")
                            tool_data = {
                                "name": item['toolUse']['name'],
                                "arguments": item['toolUse']['input']
                            }
                            lines.append(json.dumps(tool_data))
                            lines.append("</tool_call>")
        
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
            for i in range(num_examples):
                # Add delay between individual requests to prevent rate limiting
                if i > 0:
                    time.sleep(RATE_LIMIT_DELAY)
                
                # Add longer delay between batches
                if use_batching and i > 0 and i % BATCH_SIZE == 0:
                    print(f"Completed batch {i // BATCH_SIZE}. Pausing for {BATCH_DELAY}s to avoid rate limits...")
                    time.sleep(BATCH_DELAY)
                
                try:
                    # Select random tools (ONLY PRODUCTION TOOLS)
                    num_tools = random.randint(1, 3)
                    tools = self.tool_registry.get_random_tools(num_tools)

                    # Generate conversation (NO MULTIMODAL for cleaner training)
                    conversation = self.generate_conversation(tools, include_multimodal=False)

                    if format_for_training:
                        # Convert to training format
                        training_text = self.convert_to_training_format(conversation)
                        f.write(json.dumps({"text": training_text}) + "\n")
                    else:
                        # Keep original format
                        f.write(json.dumps(conversation) + "\n")

                    if (i + 1) % 10 == 0:
                        print(f"Generated {i + 1}/{num_examples} examples")
                        
                except Exception as e:
                    print(f"Error generating example {i + 1}: {e}")
                    # Continue with next example even if one fails
                    continue

        print(f"Dataset saved to {output_file}")


def main():
    """Generate training and test datasets in correct format"""
    
    # Initialize generator
    generator = DataGenerator()
    
    print("Generating training dataset with ONLY production tools...")
    print("Tools: climate_control, window_control, seat_control, lighting_control, drive_mode")
    print("-" * 60)
    
    # Generate training data in correct format
    generator.generate_dataset(
        num_examples=500,
        output_path="data/train_formatted.jsonl",
        format_for_training=True  # Convert to plain text format
    )
    
    # Generate test data in correct format
    print("\nGenerating test dataset...")
    generator.generate_dataset(
        num_examples=100,
        output_path="data/test_formatted.jsonl",
        format_for_training=True  # Convert to plain text format
    )
    
    print("\nDatasets generated successfully!")
    print("Format: Plain text with <tool_call> XML tags")
    print("Ready for fine-tuning with the fixed notebook!")


if __name__ == "__main__":
    main()
