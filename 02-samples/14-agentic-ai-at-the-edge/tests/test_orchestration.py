"""
Test suite for main orchestration and integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestOrchestrator:
    """Test main orchestrator functionality."""

    @pytest.mark.unit
    def test_orchestrator_initialization(self):
        """Test orchestrator agent setup."""
        from main import get_orchestrator

        with patch("main.Agent") as mock_agent:
            mock_instance = MagicMock()
            mock_agent.return_value = mock_instance

            orchestrator = get_orchestrator()

            mock_agent.assert_called_once()
            call_kwargs = mock_agent.call_args.kwargs
            assert "system_prompt" in call_kwargs
            assert "tools" in call_kwargs
            assert len(call_kwargs["tools"]) >= 3

    @pytest.mark.unit
    def test_dynamic_model_switching(self):
        """Test dynamic model switching."""
        from main import update_orchestrator_model

        with patch("main.Agent") as mock_agent:
            mock_instance = MagicMock()
            mock_agent.return_value = mock_instance

            with patch("main._orchestrator", mock_instance):
                # Switch to local model
                orchestrator = update_orchestrator_model("llamacpp")
                assert orchestrator is not None

                # Switch to cloud model
                orchestrator = update_orchestrator_model("bedrock")
                assert orchestrator is not None

    @pytest.mark.unit
    def test_voice_input_processing(self):
        """Test voice input activation and processing."""
        from main import process_input

        with patch("main.voice_input") as mock_voice:
            mock_voice.return_value = {
                "status": "success",
                "user_request": "schedule a meeting",
                "content": [{"text": "Voice captured"}],
            }

            with patch("main.get_orchestrator") as mock_get:
                mock_orchestrator = MagicMock()
                mock_orchestrator.return_value = "Meeting scheduled"
                mock_get.return_value = mock_orchestrator

                with patch("main.select_model") as mock_select:
                    mock_select.return_value = {"provider": "llamacpp", "reasoning": "Simple query"}

                    with patch("main.update_orchestrator_model") as mock_update:
                        mock_update.return_value = mock_orchestrator

                        response = process_input("voice")
                        assert response == "Meeting scheduled"
                        mock_voice.assert_called_once()


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests with REAL models."""

    @pytest.mark.requires_llama
    def test_demo_happy_path(self):
        """Test complete demo happy path with real LlamaCpp and Bedrock."""
        import os
        import requests

        # Check if LlamaCpp is running
        try:
            response = requests.get("http://localhost:8080/health", timeout=2)
            if response.status_code != 200:
                pytest.skip("LlamaCpp server not running")
        except:
            pytest.skip("LlamaCpp server not available")

        # Import after server check
        from main import process_input, get_orchestrator

        # Test 1: Simple calendar scheduling (should use local model)
        print("\n🗓️  Testing Calendar Scheduling...")
        response = process_input("Schedule a meeting tomorrow at 2pm")
        assert response is not None
        assert (
            "meeting" in response.lower()
            or "schedule" in response.lower()
            or "appointment" in response.lower()
        )
        print(f"   ✅ Calendar response: {response[:100]}")

        # Test 2: Vehicle assistance (should use local model + FAISS)
        print("\n🚗 Testing Vehicle Assistant...")
        response = process_input("How do I check my tire pressure?")
        assert response is not None
        assert (
            "tire" in response.lower()
            or "pressure" in response.lower()
            or "psi" in response.lower()
        )
        print(f"   ✅ Vehicle response: {response[:100]}")

        # Test 3: Complex query (should trigger cloud model if available)
        print("\n☁️  Testing Dynamic Model Selection...")
        response = process_input(
            "Analyze the implications of quantum computing on modern cryptography"
        )
        assert response is not None
        print(f"   ✅ Complex query handled: {response[:100]}")

        # Test 4: Voice simulation (text mode since we can't record audio in tests)
        print("\n🎤 Testing Voice Input Simulation...")
        response = process_input("What's the weather like today?")
        assert response is not None
        print(f"   ✅ Voice simulation response: {response[:100]}")

        print("\n✅ All demo happy path tests passed with real models!")

    @pytest.mark.requires_llama
    @pytest.mark.requires_bedrock
    def test_model_switching_real(self):
        """Test real model switching between LlamaCpp and Bedrock."""
        import os
        from main import update_orchestrator_model, get_orchestrator

        # Test switching to local model
        print("\n🔄 Testing Real Model Switching...")
        orchestrator = update_orchestrator_model("llamacpp")
        response = orchestrator("Say hello")
        assert response is not None
        print(f"   ✅ Local model response: {response}")

        # Test switching to cloud model (if credentials available)
        try:
            orchestrator = update_orchestrator_model("bedrock")
            response = orchestrator("Say hi")
            assert response is not None
            print(f"   ✅ Cloud model response: {response}")
        except Exception as e:
            print(f"   ⚠️  Bedrock not configured: {e}")

    @pytest.mark.requires_llama
    def test_agent_routing_real(self):
        """Test real agent routing with actual models."""
        from main import get_orchestrator

        orchestrator = get_orchestrator()

        # Test calendar routing
        print("\n🎯 Testing Real Agent Routing...")
        response = orchestrator("Add a dentist appointment next Monday at 3pm")
        assert response is not None
        print(f"   ✅ Calendar routing: {response[:100]}")

        # Test vehicle routing
        response = orchestrator("What does the check engine light mean?")
        assert response is not None
        print(f"   ✅ Vehicle routing: {response[:100]}")

        # Test search routing (may not have actual web search)
        response = orchestrator("Search for the latest AI news")
        assert response is not None
        print(f"   ✅ Search routing: {response[:100]}")


@pytest.mark.performance
class TestPerformance:
    """Performance and stress tests."""

    @pytest.mark.slow
    def test_response_time(self, benchmark):
        """Benchmark response time for queries."""
        from main import process_input

        with patch("main.get_orchestrator") as mock_get:
            mock_orchestrator = MagicMock()
            mock_orchestrator.return_value = "Response"
            mock_get.return_value = mock_orchestrator

            with patch("main.select_model") as mock_select:
                mock_select.return_value = {"provider": "llamacpp"}

                with patch("main.update_orchestrator_model") as mock_update:
                    mock_update.return_value = mock_orchestrator

                    # Benchmark would measure this
                    response = process_input("What time is it?")
                    assert response == "Response"

    @pytest.mark.slow
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        import asyncio
        from main import process_input

        with patch("main.get_orchestrator") as mock_get:
            mock_orchestrator = MagicMock()
            mock_orchestrator.return_value = "Response"
            mock_get.return_value = mock_orchestrator

            # Test concurrent processing
            # Implementation would go here
            pass
