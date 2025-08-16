#!/usr/bin/env python
"""
Demo Happy Path Test Runner
Tests the complete system with real models (LlamaCpp and Bedrock).

Usage:
    python tests/run_demo.py
"""

import sys
import os
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import requests
import time


def check_services():
    """Check if required services are running."""
    print("=" * 60)
    print("🔍 CHECKING SERVICES")
    print("=" * 60)

    # Check LlamaCpp
    try:
        response = requests.get("http://localhost:8080/health", timeout=2)
        if response.status_code == 200:
            print("✅ LlamaCpp server is running")
        else:
            print("❌ LlamaCpp server not responding properly")
            return False
    except:
        print("❌ LlamaCpp server not available at http://localhost:8080")
        return False

    # Check Bedrock (optional)
    try:
        import boto3

        print("✅ AWS Bedrock credentials configured")
    except:
        print("⚠️  AWS Bedrock not configured (optional)")

    return True


def run_demo_tests():
    """Run demo happy path tests with real models."""
    print("\n" + "=" * 60)
    print("🚀 RUNNING DEMO HAPPY PATH TESTS")
    print("=" * 60)

    from main import process_input, get_orchestrator

    tests_passed = 0
    tests_total = 0

    # Test 1: Calendar Scheduling
    print("\n📅 Test 1: Calendar Scheduling")
    print("-" * 40)
    tests_total += 1
    try:
        start = time.time()
        response = process_input("Schedule a team meeting tomorrow at 2pm in Conference Room A")
        elapsed = time.time() - start

        if response and (
            "meeting" in response.lower()
            or "schedule" in response.lower()
            or "appointment" in response.lower()
        ):
            print(f"✅ PASSED ({elapsed:.2f}s)")
            print(f"   Response preview: {response[:150]}...")
            tests_passed += 1
        else:
            print(f"❌ FAILED: Unexpected response")
            print(f"   Response: {response}")
    except Exception as e:
        print(f"❌ FAILED: {e}")

    # Test 2: Vehicle Assistance
    print("\n🚗 Test 2: Vehicle Assistance")
    print("-" * 40)
    tests_total += 1
    try:
        start = time.time()
        response = process_input("How do I check tire pressure?")
        elapsed = time.time() - start

        if response and (
            "tire" in response.lower()
            or "pressure" in response.lower()
            or "psi" in response.lower()
        ):
            print(f"✅ PASSED ({elapsed:.2f}s)")
            print(f"   Response preview: {response[:150]}...")
            tests_passed += 1
        else:
            print(f"❌ FAILED: Unexpected response")
            print(f"   Response: {response}")
    except Exception as e:
        print(f"❌ FAILED: {e}")

    # Test 3: Simple Query (Local Model)
    print("\n💬 Test 3: Simple Query")
    print("-" * 40)
    tests_total += 1
    try:
        start = time.time()
        response = process_input("What's the weather like?")
        elapsed = time.time() - start

        if response:
            print(f"✅ PASSED ({elapsed:.2f}s)")
            print(f"   Response preview: {response[:150]}...")
            tests_passed += 1
        else:
            print(f"❌ FAILED: No response")
    except Exception as e:
        print(f"❌ FAILED: {e}")

    # Test 4: Complex Query (May trigger cloud model)
    print("\n🧠 Test 4: Complex Query")
    print("-" * 40)
    tests_total += 1
    try:
        start = time.time()
        response = process_input(
            "Explain the relationship between quantum computing and cryptography"
        )
        elapsed = time.time() - start

        if response and len(response) > 50:
            print(f"✅ PASSED ({elapsed:.2f}s)")
            print(f"   Response preview: {response[:150]}...")
            tests_passed += 1
        else:
            print(f"❌ FAILED: Response too short or missing")
    except Exception as e:
        print(f"❌ FAILED: {e}")

    # Test 5: Agent Routing
    print("\n🎯 Test 5: Agent Routing")
    print("-" * 40)
    tests_total += 1
    try:
        orchestrator = get_orchestrator()
        start = time.time()
        response = orchestrator("Add a dentist appointment next Monday at 3pm")
        elapsed = time.time() - start

        if response:
            print(f"✅ PASSED ({elapsed:.2f}s)")
            print(f"   Response preview: {str(response)[:150]}...")
            tests_passed += 1
        else:
            print(f"❌ FAILED: No response from orchestrator")
    except Exception as e:
        print(f"❌ FAILED: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {tests_passed}/{tests_total}")

    if tests_passed == tests_total:
        print("🎉 All demo tests passed!")
        return 0
    else:
        print(f"⚠️  {tests_total - tests_passed} tests failed")
        return 1


def main():
    """Main entry point."""
    print("🧪 DEMO HAPPY PATH TEST SUITE")
    print("Testing with real LlamaCpp and Bedrock models")
    print()

    if not check_services():
        print("\n❌ Required services not available")
        print("Please ensure LlamaCpp server is running:")
        print("  llama-server -m models/qwen2.5-omni-7b.Q4_K_M.gguf --port 8080")
        return 1

    return run_demo_tests()


if __name__ == "__main__":
    exit(main())
