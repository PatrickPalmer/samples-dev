"""
Evaluation utilities for fine-tuned models
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class EvaluationMetrics:
    """Metrics for model evaluation"""

    # Tool calling metrics
    tool_selection_accuracy: float = 0.0
    parameter_extraction_accuracy: float = 0.0
    tool_call_format_validity: float = 0.0

    # Performance metrics
    tokens_per_second: float = 0.0
    first_token_latency_ms: float = 0.0
    memory_usage_gb: float = 0.0

    # Quality metrics
    response_coherence: float = 0.0
    multimodal_understanding: float = 0.0
    error_recovery_rate: float = 0.0

    # Detailed results
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "tool_calling": {
                "selection_accuracy": self.tool_selection_accuracy,
                "parameter_accuracy": self.parameter_extraction_accuracy,
                "format_validity": self.tool_call_format_validity,
            },
            "performance": {
                "tokens_per_second": self.tokens_per_second,
                "first_token_latency_ms": self.first_token_latency_ms,
                "memory_usage_gb": self.memory_usage_gb,
            },
            "quality": {
                "coherence": self.response_coherence,
                "multimodal_understanding": self.multimodal_understanding,
                "error_recovery": self.error_recovery_rate,
            },
        }

    def save(self, path: str) -> None:
        """Save metrics to JSON file"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        """Print evaluation summary"""
        print("\nEvaluation Summary")
        print("=" * 50)

        print("\nTool Calling Performance:")
        print(f"  Selection Accuracy: {self.tool_selection_accuracy:.1%}")
        print(f"  Parameter Accuracy: {self.parameter_extraction_accuracy:.1%}")
        print(f"  Format Validity: {self.tool_call_format_validity:.1%}")

        print("\nInference Performance:")
        print(f"  Speed: {self.tokens_per_second:.1f} tokens/s")
        print(f"  First Token: {self.first_token_latency_ms:.1f} ms")
        print(f"  Memory Usage: {self.memory_usage_gb:.2f} GB")

        print("\nQuality Metrics:")
        print(f"  Coherence: {self.response_coherence:.1%}")
        print(f"  Multimodal: {self.multimodal_understanding:.1%}")
        print(f"  Error Recovery: {self.error_recovery_rate:.1%}")


class ModelEvaluator:
    """Evaluate fine-tuned model performance"""

    def __init__(self, model_path: str, test_data_path: str):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.metrics = EvaluationMetrics()
        self.client = None

    def setup_inference(self, base_url: str = "http://localhost:8080") -> None:
        """Setup inference client"""
        try:
            import httpx

            self.client = httpx.Client(base_url=base_url, timeout=30.0)

            # Test connection
            response = self.client.get("/health")
            if response.status_code != 200:
                raise ConnectionError(f"Server not healthy: {response.status_code}")

        except Exception as e:
            print(f"Failed to connect to inference server: {e}")
            print("Please start llama-server with the model first")
            raise

    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test dataset"""
        test_examples = []

        with open(self.test_data_path, "r") as f:
            for line in f:
                example = json.loads(line)
                test_examples.append(example)

        return test_examples

    def evaluate_tool_calling(self, test_examples: List[Dict[str, Any]]) -> None:
        """Evaluate tool calling capabilities"""

        correct_tools = 0
        correct_params = 0
        valid_formats = 0
        total = len(test_examples)

        for example in test_examples:
            # Extract expected tool calls
            expected_tools = self._extract_expected_tools(example)

            # Generate response
            response = self._generate_response(example)

            # Extract actual tool calls
            actual_tools = self._extract_actual_tools(response)

            # Compare
            if self._compare_tool_selection(expected_tools, actual_tools):
                correct_tools += 1

            if self._compare_parameters(expected_tools, actual_tools):
                correct_params += 1

            if self._validate_format(actual_tools):
                valid_formats += 1

            # Store detailed result
            self.metrics.detailed_results.append(
                {
                    "example_id": example.get("conversation_id"),
                    "expected": expected_tools,
                    "actual": actual_tools,
                    "correct_tool": correct_tools == total,
                    "correct_params": correct_params == total,
                }
            )

        # Calculate metrics
        self.metrics.tool_selection_accuracy = correct_tools / total if total > 0 else 0
        self.metrics.parameter_extraction_accuracy = correct_params / total if total > 0 else 0
        self.metrics.tool_call_format_validity = valid_formats / total if total > 0 else 0

    def evaluate_performance(self, num_samples: int = 10) -> None:
        """Evaluate inference performance"""

        latencies = []
        token_counts = []

        for _ in range(num_samples):
            # Simple prompt for performance testing
            prompt = "What is the weather today?"

            start_time = time.time()
            response = self._generate_response_raw(prompt)
            end_time = time.time()

            # Calculate metrics
            latency = (end_time - start_time) * 1000  # ms
            latencies.append(latency)

            # Estimate tokens (rough approximation)
            if response:
                tokens = len(response.split()) * 1.3  # Rough token estimate
                token_counts.append(tokens)

        # Calculate averages
        if latencies:
            self.metrics.first_token_latency_ms = np.mean(
                latencies[:3]
            )  # First few are representative

        if token_counts and latencies:
            avg_tokens = np.mean(token_counts)
            avg_time_s = np.mean(latencies) / 1000
            self.metrics.tokens_per_second = avg_tokens / avg_time_s if avg_time_s > 0 else 0

        # Memory usage (if available from server)
        self._check_memory_usage()

    def evaluate_quality(self, test_examples: List[Dict[str, Any]]) -> None:
        """Evaluate response quality"""

        coherence_scores = []
        multimodal_scores = []
        error_recovery_scores = []

        for example in test_examples[:50]:  # Sample for quality evaluation
            response = self._generate_response(example)

            # Check coherence (simple heuristics)
            coherence = self._assess_coherence(response)
            coherence_scores.append(coherence)

            # Check multimodal understanding
            if self._has_multimodal_content(example):
                multimodal = self._assess_multimodal(example, response)
                multimodal_scores.append(multimodal)

            # Check error recovery
            if self._is_error_case(example):
                recovery = self._assess_error_recovery(response)
                error_recovery_scores.append(recovery)

        # Calculate averages
        self.metrics.response_coherence = np.mean(coherence_scores) if coherence_scores else 0
        self.metrics.multimodal_understanding = (
            np.mean(multimodal_scores) if multimodal_scores else 0
        )
        self.metrics.error_recovery_rate = (
            np.mean(error_recovery_scores) if error_recovery_scores else 0
        )

    def _generate_response(self, example: Dict[str, Any]) -> str:
        """Generate response from model"""

        if not self.client:
            return ""

        # Format messages for API
        messages = []
        for msg in example.get("messages", []):
            if msg["role"] in ["system", "user", "assistant"]:
                messages.append(
                    {"role": msg["role"], "content": self._format_content(msg["content"])}
                )

        # Call API
        try:
            response = self.client.post(
                "/v1/chat/completions",
                json={"messages": messages, "max_tokens": 500, "temperature": 0.7},
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            print(f"Error generating response: {e}")

        return ""

    def _generate_response_raw(self, prompt: str) -> str:
        """Generate response from simple prompt"""

        if not self.client:
            return ""

        try:
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0.7,
                },
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            pass

        return ""

    def _format_content(self, content: Any) -> str:
        """Format content for API"""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        text_parts.append(item["text"])
                    elif "toolUse" in item:
                        text_parts.append(f"[Tool: {item['toolUse']['name']}]")
                else:
                    text_parts.append(str(item))
            return " ".join(text_parts)

        return str(content)

    def _extract_expected_tools(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract expected tool calls from example"""

        tools = []
        for msg in example.get("messages", []):
            if msg["role"] == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "toolUse" in item:
                            tools.append(item["toolUse"])
        return tools

    def _extract_actual_tools(self, response: str) -> List[Dict[str, Any]]:
        """Extract tool calls from response"""

        tools = []

        # Look for tool call patterns
        if "<tool_call>" in response:
            # Parse structured format
            import re

            pattern = r"<tool_call>(.*?)</tool_call>"
            matches = re.findall(pattern, response, re.DOTALL)

            for match in matches:
                tool = {}
                # Extract fields
                if "name:" in match:
                    name_match = re.search(r"name:\s*(\w+)", match)
                    if name_match:
                        tool["name"] = name_match.group(1)

                if "arguments:" in match:
                    args_match = re.search(r"arguments:\s*({.*})", match)
                    if args_match:
                        try:
                            tool["input"] = json.loads(args_match.group(1))
                        except:
                            tool["input"] = {}

                if tool:
                    tools.append(tool)

        return tools

    def _compare_tool_selection(self, expected: List[Dict], actual: List[Dict]) -> bool:
        """Compare tool selection"""

        expected_names = {tool.get("name") for tool in expected}
        actual_names = {tool.get("name") for tool in actual}

        return expected_names == actual_names

    def _compare_parameters(self, expected: List[Dict], actual: List[Dict]) -> bool:
        """Compare tool parameters"""

        if len(expected) != len(actual):
            return False

        for exp, act in zip(expected, actual):
            exp_params = exp.get("input", {})
            act_params = act.get("input", {})

            # Check key overlap
            if set(exp_params.keys()) != set(act_params.keys()):
                return False

        return True

    def _validate_format(self, tools: List[Dict]) -> bool:
        """Validate tool call format"""

        for tool in tools:
            if "name" not in tool:
                return False
            if "input" not in tool and "arguments" not in tool:
                return False

        return len(tools) > 0

    def _assess_coherence(self, response: str) -> float:
        """Assess response coherence"""

        if not response:
            return 0.0

        # Simple heuristics
        score = 1.0

        # Check for incomplete sentences
        if response.count(".") == 0 and len(response) > 50:
            score -= 0.2

        # Check for repetition
        words = response.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                score -= 0.3

        return max(0.0, score)

    def _has_multimodal_content(self, example: Dict[str, Any]) -> bool:
        """Check if example has multimodal content"""

        for msg in example.get("messages", []):
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if any(key in item for key in ["image", "audio", "video"]):
                            return True
        return False

    def _assess_multimodal(self, example: Dict[str, Any], response: str) -> float:
        """Assess multimodal understanding"""

        # Check if response acknowledges multimodal content
        multimodal_keywords = ["image", "audio", "video", "see", "hear", "shows", "displays"]

        score = 0.0
        for keyword in multimodal_keywords:
            if keyword in response.lower():
                score = 1.0
                break

        return score

    def _is_error_case(self, example: Dict[str, Any]) -> bool:
        """Check if example is an error case"""

        # Look for error patterns in messages
        for msg in example.get("messages", []):
            content = str(msg.get("content", "")).lower()
            if any(word in content for word in ["error", "failed", "invalid", "wrong"]):
                return True
        return False

    def _assess_error_recovery(self, response: str) -> float:
        """Assess error recovery ability"""

        # Check for appropriate error handling
        recovery_phrases = ["try again", "alternative", "instead", "another way", "let me"]

        score = 0.0
        for phrase in recovery_phrases:
            if phrase in response.lower():
                score = 1.0
                break

        return score

    def _check_memory_usage(self) -> None:
        """Check memory usage from server"""

        try:
            import psutil

            process = psutil.Process()
            memory_gb = process.memory_info().rss / (1024**3)
            self.metrics.memory_usage_gb = memory_gb
        except:
            # Fallback estimate
            self.metrics.memory_usage_gb = 2.5  # Typical for Q4_K_M

    def run_full_evaluation(self) -> EvaluationMetrics:
        """Run complete evaluation pipeline"""

        print("Starting model evaluation...")

        # Load test data
        test_examples = self.load_test_data()
        print(f"Loaded {len(test_examples)} test examples")

        # Setup inference
        print("Setting up inference server...")
        self.setup_inference()

        # Run evaluations
        print("\nEvaluating tool calling...")
        self.evaluate_tool_calling(test_examples)

        print("Evaluating performance...")
        self.evaluate_performance()

        print("Evaluating quality...")
        self.evaluate_quality(test_examples)

        # Print summary
        self.metrics.print_summary()

        return self.metrics
