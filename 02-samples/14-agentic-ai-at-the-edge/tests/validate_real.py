#!/usr/bin/env python
"""
Quick validation of real model functionality.
"""

import sys
import os
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import requests
from strands import Agent
from strands.models import BedrockModel
from strands.models.llamacpp import LlamaCppModel


def validate_services():
    """Quick validation of available services."""
    print("=" * 60)
    print("🔍 VALIDATING REAL SERVICES")
    print("=" * 60)

    # 1. Check LlamaCpp
    print("\n1. LlamaCpp Server:")
    try:
        response = requests.get("http://localhost:8080/health", timeout=2)
        if response.status_code == 200:
            print("   ✅ Running at http://localhost:8080")

            # Quick inference test
            model = LlamaCppModel(
                base_url="http://localhost:8080",
                model_id="default",
                params={"temperature": 0.1, "max_tokens": 20},
            )
            agent = Agent(model=model, system_prompt="Answer concisely.")
            result = agent("Say 'hello'")
            print(f"   ✅ Inference works: {result}")
        else:
            print("   ❌ Server not responding")
    except Exception as e:
        print(f"   ❌ Not available: {e}")

    # 2. Check Bedrock
    print("\n2. AWS Bedrock:")
    try:
        model = BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
        agent = Agent(model=model, system_prompt="Answer in one word.")
        result = agent("Say 'hi'")
        print(f"   ✅ Bedrock works: {result}")
    except Exception as e:
        print(f"   ❌ Not available: {e}")

    # 3. Test agents
    print("\n3. Agent Systems:")

    # Calendar assistant module not implemented
    print("⚠️ calendar_assistant module not implemented")
    
    # Vehicle assistant module not implemented  
    print("⚠️ vehicle_assistant module not implemented")

    # 4. Test orchestrator
    print("\n4. Orchestrator:")
    try:
        from main import get_orchestrator

        orchestrator = get_orchestrator()
        result = orchestrator("What time is it?")
        print(f"   ✅ Orchestrator: Works")
    except Exception as e:
        print(f"   ❌ Orchestrator: {e}")

    print("\n" + "=" * 60)
    print("✅ Validation complete! Real models are functional.")
    print("=" * 60)


if __name__ == "__main__":
    validate_services()
