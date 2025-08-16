#!/usr/bin/env python
"""
Simple integration tests using real models.
Run this to validate actual functionality (not mocked).

Usage:
    python tests/test_real_models.py
"""

import sys
import os
from pathlib import Path

# Add both src and parent to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import requests
from strands import Agent, tool
from strands.models import LlamaCppModel, BedrockModel


def test_llamacpp_server():
    """Test if LlamaCpp server is available."""
    print("\n🔍 Testing LlamaCpp server...")
    try:
        response = requests.get("http://localhost:8080/health", timeout=2)
        if response.status_code == 200:
            print("✅ LlamaCpp server is running")
            return True
    except:
        print("❌ LlamaCpp server not available at http://localhost:8080")
        return False
    return False


def test_real_llamacpp_model():
    """Test actual LlamaCpp model inference."""
    print("\n🧠 Testing LlamaCpp model inference...")
    try:
        model = LlamaCppModel(
            base_url="http://localhost:8080",
            model_id="default",
            params={"temperature": 0.7, "max_tokens": 100},
        )
        agent = Agent(model=model, system_prompt="You are a helpful assistant. Be concise.")

        # Test basic math
        response = agent("What is 15 + 27? Just give the number.")
        print(f"  Math test: 15 + 27 = {response}")
        assert "42" in str(response), f"Expected 42, got {response}"

        print("✅ LlamaCpp model inference works")
        return True
    except Exception as e:
        print(f"❌ LlamaCpp model failed: {e}")
        return False


def test_real_bedrock_model():
    """Test actual Bedrock model inference."""
    print("\n☁️  Testing Bedrock model...")
    try:
        model = BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
        agent = Agent(model=model, system_prompt="You are a helpful assistant. Be concise.")

        # Test geography
        response = agent("What is the capital of Japan? One word only.")
        print(f"  Geography test: Capital of Japan = {response}")
        assert "Tokyo" in str(response), f"Expected Tokyo, got {response}"

        print("✅ Bedrock model works")
        return True
    except Exception as e:
        print(f"❌ Bedrock model failed: {e}")
        return False


def test_calendar_agent():
    """Test calendar agent with real model."""
    print("\n📅 Testing Calendar Agent...")
    try:
        # Module doesn't exist - skip this test
        print("⚠️ calendar_assistant module not implemented - skipping")
        return False
    except Exception as e:
        print(f"❌ Calendar agent failed: {e}")
        return False


def test_vehicle_agent():
    """Test vehicle agent with real FAISS."""
    print("\n🚗 Testing Vehicle Agent...")
    try:
        # Module doesn't exist - skip this test
        print("⚠️ vehicle_assistant module not implemented - skipping")
        return False
    except Exception as e:
        print(f"❌ Vehicle agent failed: {e}")
        return False


def test_model_selector():
    """Test model selector logic."""
    print("\n🔄 Testing Model Selector...")
    try:
        from src.agents.tools.model_selector import select_model

        # Simple query
        result = select_model("What time is it?")
        print(f"  Simple query -> {result['provider']} (local expected)")
        assert result["provider"] == "llamacpp"

        # Complex query
        result = select_model("Analyze quantum computing implications for cryptography")
        print(f"  Complex query -> {result['provider']} (cloud expected)")
        assert result["provider"] == "bedrock"

        print("✅ Model selector works")
        return True
    except Exception as e:
        print(f"❌ Model selector failed: {e}")
        return False


def test_tool_calling():
    """Test tool calling with real model."""
    print("\n🔧 Testing Tool Calling...")
    try:

        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"The weather in {city} is sunny and 72°F"

        model = LlamaCppModel(base_url="http://localhost:8080", model_id="default")
        agent = Agent(
            model=model, tools=[get_weather], system_prompt="You are a weather assistant."
        )

        response = agent("What's the weather in Paris?")
        response_str = str(response)
        print(f"  Tool call response: {response_str[:100]}...")
        assert "Paris" in response_str or "weather" in response_str.lower()

        print("✅ Tool calling works")
        return True
    except Exception as e:
        print(f"❌ Tool calling failed: {e}")
        return False


def test_orchestrator():
    """Test main orchestrator with real models."""
    print("\n🎭 Testing Orchestrator...")
    try:
        from main import get_orchestrator

        orchestrator = get_orchestrator()

        # Test calendar routing
        response = orchestrator("Schedule a meeting at 2pm")
        print(f"  Calendar routing: {response[:50]}...")

        # Test vehicle routing
        response = orchestrator("What's the oil change interval?")
        print(f"  Vehicle routing: {response[:50]}...")

        print("✅ Orchestrator works")
        return True
    except Exception as e:
        print(f"❌ Orchestrator failed: {e}")
        return False


# Image analysis test removed - functionality deprecated


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("🧪 INTEGRATION TESTS WITH REAL MODELS")
    print("=" * 60)

    results = []

    # Check server availability
    if test_llamacpp_server():
        results.append(("LlamaCpp Server", test_llamacpp_server()))
        results.append(("LlamaCpp Model", test_real_llamacpp_model()))
        results.append(("Tool Calling", test_tool_calling()))
        results.append(("Model Selector", test_model_selector()))
        results.append(("Calendar Agent", test_calendar_agent()))
        results.append(("Vehicle Agent", test_vehicle_agent()))
        results.append(("Orchestrator", test_orchestrator()))
        # Image analysis removed - functionality deprecated

    # Test Bedrock if available
    try:
        import boto3

        results.append(("Bedrock Model", test_real_bedrock_model()))
    except:
        print("\n⚠️  Skipping Bedrock tests (credentials not configured)")

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("🎉 All integration tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
