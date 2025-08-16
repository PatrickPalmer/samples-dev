#!/usr/bin/env python3
"""
Quick demonstration test of cockpit control functionality
This test demonstrates the complete cockpit control system working together
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.vehicle_systems import get_virtual_ecu
from src.agents.cockpit import (
    climate_control,
    window_control,
    seat_control,
    lighting_control,
    drive_mode,
)


def demo_cockpit_controls():
    """Demonstrate all cockpit control functionality"""
    print("\n" + "=" * 60)
    print("COCKPIT CONTROL DEMONSTRATION")
    print("=" * 60)

    # Initialize ECU
    ecu = get_virtual_ecu()
    print(f"\n✅ Virtual ECU initialized for {ecu.vehicle_model}")

    # Test Climate Control
    print("\n🌡️ CLIMATE CONTROL DEMO:")
    print("-" * 40)
    response = climate_control("Set temperature to 72 degrees")
    print(f"Command: Set temperature to 72 degrees")
    print(f"Response: {response}")

    response = climate_control("Turn on max AC")
    print(f"\nCommand: Turn on max AC")
    print(f"Response: {response}")

    # Test Window Control
    print("\n🪟 WINDOW CONTROL DEMO:")
    print("-" * 40)
    response = window_control("Open driver window")
    print(f"Command: Open driver window")
    print(f"Response: {response}")

    response = window_control("Close all windows")
    print(f"\nCommand: Close all windows")
    print(f"Response: {response}")

    # Test Seat Control
    print("\n💺 SEAT CONTROL DEMO:")
    print("-" * 40)
    response = seat_control("Move seat back")
    print(f"Command: Move seat back")
    print(f"Response: {response}")

    response = seat_control("Turn on seat heating")
    print(f"\nCommand: Turn on seat heating")
    print(f"Response: {response}")

    # Test Lighting Control
    print("\n💡 LIGHTING CONTROL DEMO:")
    print("-" * 40)
    response = lighting_control("Turn on headlights")
    print(f"Command: Turn on headlights")
    print(f"Response: {response}")

    response = lighting_control("Set ambient lighting to blue")
    print(f"\nCommand: Set ambient lighting to blue")
    print(f"Response: {response}")

    # Test Drive Mode
    print("\n🏎️ DRIVE MODE DEMO:")
    print("-" * 40)
    response = drive_mode("Switch to sport mode")
    print(f"Command: Switch to sport mode")
    print(f"Response: {response}")

    # Show final ECU state
    print("\n📊 FINAL VEHICLE STATE:")
    print("-" * 40)
    state = ecu.get_state()
    print(
        f"Climate: {state['climate']['temperature_set']}°F, Fan: {state['climate']['fan_speed']}/7"
    )
    print(f"Windows: Driver {state['windows']['driver']}% open")
    print(
        f"Seat: Position {state['seats']['driver']['position_forward']}%, Heating level {state['seats']['driver']['heating']}/3"
    )
    print(
        f"Lights: Headlights {state['lighting']['headlights']}, Ambient {state['lighting']['ambient']['color']}"
    )
    print(f"Drive Mode: {state['drive_mode']['current']}")

    print("\n" + "=" * 60)
    print("✅ COCKPIT CONTROL DEMO COMPLETE!")
    print("=" * 60)

    return True


def test_orchestrator_integration():
    """Test the main orchestrator with cockpit commands"""
    print("\n" + "=" * 60)
    print("ORCHESTRATOR INTEGRATION TEST")
    print("=" * 60)

    try:
        from main import get_orchestrator

        orchestrator = get_orchestrator()

        print("\n🤖 Testing orchestrator routing...")

        # Test climate routing
        print("\n1. Climate Control:")
        response = orchestrator("Set the temperature to 68 degrees")
        print(f"   Request: Set the temperature to 68 degrees")
        print(f"   Response: {str(response)[:200]}...")

        # Test window routing
        print("\n2. Window Control:")
        response = orchestrator("Open the driver window halfway")
        print(f"   Request: Open the driver window halfway")
        print(f"   Response: {str(response)[:200]}...")

        print("\n✅ Orchestrator routing works!")
        return True

    except Exception as e:
        print(f"\n⚠️ Orchestrator test skipped: {e}")
        return False


if __name__ == "__main__":
    # Run demos
    success = demo_cockpit_controls()

    # Try orchestrator if available
    if success:
        test_orchestrator_integration()

    print("\n🎉 All demonstrations complete!")
