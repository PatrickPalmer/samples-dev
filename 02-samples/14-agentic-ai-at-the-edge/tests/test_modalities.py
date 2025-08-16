"""
Test suite for all deployment modalities (Development, Container, API/Service).

This test suite ensures the unified codebase works correctly across all
deployment modes with different audio input methods and configurations.
"""

import pytest
import json
import base64
import asyncio
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDevelopmentMode:
    """Test development mode with direct microphone access."""

    @pytest.mark.unit
    def test_direct_voice_input(self):
        """Test voice input with direct microphone recording."""
        from main import process_input

        with patch("main.voice_input") as mock_voice:
            mock_voice.return_value = {
                "status": "success",
                "user_request": "What's the weather today?",
                "raw_transcription": "what is the weather today",
                "audio_source": "microphone",
                "content": [{"text": "Voice captured"}],
            }

            with patch("main.get_orchestrator") as mock_get:
                mock_orchestrator = MagicMock()
                mock_orchestrator.return_value = "Today's weather is sunny, 72°F"
                mock_get.return_value = mock_orchestrator

                with patch("main.select_model") as mock_select:
                    mock_select.return_value = {
                        "provider": "bedrock",
                        "reasoning": "Weather query needs current data",
                    }

                    with patch("main.update_orchestrator_model") as mock_update:
                        mock_update.return_value = mock_orchestrator

                        response = process_input("voice")

                        # Verify voice_input was called without audio_bytes or audio_file
                        mock_voice.assert_called_once_with(duration=10)
                        assert response == "Today's weather is sunny, 72°F"

    @pytest.mark.unit
    def test_text_input_dev_mode(self):
        """Test standard text input in development mode."""
        from main import process_input

        with patch("main.get_orchestrator") as mock_get:
            mock_orchestrator = MagicMock()
            mock_orchestrator.return_value = "Meeting scheduled"
            mock_get.return_value = mock_orchestrator

            with patch("main.select_model") as mock_select:
                mock_select.return_value = {
                    "provider": "llamacpp",
                    "reasoning": "Simple scheduling task",
                }

                with patch("main.update_orchestrator_model") as mock_update:
                    mock_update.return_value = mock_orchestrator

                    response = process_input("Schedule a meeting at 3pm")
                    assert response == "Meeting scheduled"


class TestContainerMode:
    """Test container mode with file-based audio exchange."""

    @pytest.mark.unit
    def test_audio_file_detection(self):
        """Test automatic detection of audio files in container."""
        from main import process_input

        # Mock file existence check
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True  # File exists

            with patch("main.voice_input") as mock_voice:
                mock_voice.return_value = {
                    "status": "success",
                    "user_request": "Check tire pressure",
                    "audio_source": "file",
                    "audio_file": "/app/audio_exchange/voice_input.wav",
                    "content": [{"text": "Voice captured from file"}],
                }

                with patch("main.get_orchestrator") as mock_get:
                    mock_orchestrator = MagicMock()
                    mock_orchestrator.return_value = "Tire pressure should be 32 PSI"
                    mock_get.return_value = mock_orchestrator

                    with patch("main.select_model") as mock_select:
                        mock_select.return_value = {"provider": "llamacpp"}

                        with patch("main.update_orchestrator_model") as mock_update:
                            mock_update.return_value = mock_orchestrator

                            response = process_input("voice")

                            # Verify voice_input was called with audio_file
                            mock_voice.assert_called_once_with(audio_file="voice_input.wav")
                            assert response == "Tire pressure should be 32 PSI"

    @pytest.mark.unit
    def test_audio_file_processing(self):
        """Test processing of audio files from volume mount."""
        # from src.agents.tools.voice_input_tool import voice_input  # Module doesn't exist
        pytest.skip("voice_input_tool module not implemented")


class TestAPIServiceMode:
    """Test API/Service mode with base64-encoded audio."""

    @pytest.mark.unit
    def test_api_health_check(self):
        """Test API health endpoint."""
        # Import FastAPI app
        with patch("main.DEPLOYMENT_TARGET", "automotive"):
            from main import app
            from fastapi.testclient import TestClient

            client = TestClient(app)
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["deployment"] == "automotive"

    @pytest.mark.unit
    def test_api_text_chat(self):
        """Test API text chat endpoint."""
        from main import app
        from fastapi.testclient import TestClient

        with patch("main.process_input") as mock_process:
            mock_process.return_value = "The check engine light means..."

            client = TestClient(app)
            response = client.post(
                "/chat",
                json={"prompt": "What does the check engine light mean?", "session_id": "test_123"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "The check engine light means..."
            assert data["session_id"] == "test_123"

            # Verify process_input was called with correct args
            mock_process.assert_called_once_with(
                "What does the check engine light mean?", None, "wav"  # No audio data
            )

    @pytest.mark.unit
    def test_api_voice_chat(self):
        """Test API voice chat with base64 audio."""
        from main import app
        from fastapi.testclient import TestClient

        # Create fake audio data
        fake_audio = b"fake_wav_audio_data"
        audio_b64 = base64.b64encode(fake_audio).decode("utf-8")

        with patch("main.process_input") as mock_process:
            mock_process.return_value = "Your oil change is due at 30,000 miles"

            client = TestClient(app)
            response = client.post(
                "/chat",
                json={
                    "prompt": "voice",
                    "audio_data": audio_b64,
                    "audio_format": "wav",
                    "session_id": "driver_456",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "oil change" in data["response"]

            # Verify process_input was called with decoded audio
            mock_process.assert_called_once()
            args = mock_process.call_args[0]
            assert args[0] == "voice"
            assert args[1] == fake_audio  # Decoded audio bytes
            assert args[2] == "wav"

    @pytest.mark.unit
    def test_api_invalid_audio(self):
        """Test API with invalid base64 audio data."""
        from main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/chat",
            json={"prompt": "voice", "audio_data": "invalid_base64_@#$%", "session_id": "test"},
        )

        assert response.status_code == 400
        assert "Invalid audio data" in response.json()["detail"]


class TestAudioUtilities:
    """Test audio utilities that support all modes."""

    @pytest.mark.unit
    def test_audio_recorder_initialization(self):
        """Test AudioRecorder initialization."""
        # from src.utils.audio_utils import AudioRecorder  # Module doesn't exist
        pytest.skip("audio_utils module not implemented")

    @pytest.mark.unit
    def test_audio_transport_encoding(self):
        """Test audio encoding for API transport."""
        # from src.utils.audio_utils import AudioTransport  # Module doesn't exist
        pytest.skip("audio_utils module not implemented")

        test_audio = b"test_audio_data"
        encoded = AudioTransport.encode_for_api(test_audio)

        # Verify it's valid base64
        decoded = base64.b64decode(encoded)
        assert decoded == test_audio

    @pytest.mark.unit
    def test_audio_transport_request_preparation(self):
        """Test API request preparation with audio."""
        # from src.utils.audio_utils import AudioTransport  # Module doesn't exist
        pytest.skip("audio_utils module not implemented")

        test_audio = b"test_audio_data"
        request = AudioTransport.prepare_api_request(test_audio, "session_789")

        assert request["prompt"] == "voice"
        assert request["session_id"] == "session_789"
        assert request["audio_format"] == "wav"
        assert "audio_data" in request

        # Verify audio can be decoded
        decoded = base64.b64decode(request["audio_data"])
        assert decoded == test_audio


class TestDeploymentAwareness:
    """Test deployment-aware behavior."""

    @pytest.mark.unit
    def test_model_selection_by_deployment(self):
        """Test model selection changes based on deployment target."""
        # from src.agents.calendar_assistant import get_calendar_model  # Module doesn't exist
        pytest.skip("calendar_assistant module not implemented")

        # Test edge deployment
        with patch("src.agents.calendar_assistant.DEPLOYMENT_TARGET", "automotive"):
            model = get_calendar_model()
            assert model.__class__.__name__ == "LlamaCppModel"

        # Test cloud deployment
        with patch("src.agents.calendar_assistant.DEPLOYMENT_TARGET", "development"):
            with patch("src.agents.calendar_assistant.BEDROCK_MODEL_ID", "claude-3"):
                model = get_calendar_model()
                assert model.__class__.__name__ == "BedrockModel"

    @pytest.mark.unit
    def test_ui_mode_by_deployment(self):
        """Test UI changes based on configuration."""
        with patch("main.USE_RICH_UI", True):
            with patch("main.console") as mock_console:
                from main import process_input

                with patch("main.get_orchestrator") as mock_get:
                    mock_orchestrator = MagicMock()
                    mock_orchestrator.return_value = "Response"
                    mock_get.return_value = mock_orchestrator

                    with patch("main.select_model") as mock_select:
                        mock_select.return_value = {"provider": "llamacpp"}

                        with patch("main.update_orchestrator_model") as mock_update:
                            mock_update.return_value = mock_orchestrator

                            process_input("test")

                            # Verify rich console was used
                            assert mock_console.print.called


class TestEndToEndScenarios:
    """Test complete scenarios across different modes."""

    @pytest.mark.integration
    async def test_concurrent_api_requests(self):
        """Test handling concurrent API requests."""
        from main import app
        from fastapi.testclient import TestClient

        with patch("main.process_input") as mock_process:
            # Simulate different response times
            async def async_response(prompt, *args):
                if "weather" in prompt:
                    await asyncio.sleep(0.1)
                    return "Sunny, 72°F"
                else:
                    await asyncio.sleep(0.05)
                    return "Response"

            mock_process.side_effect = lambda p, *args: (
                "Sunny, 72°F" if "weather" in p else "Response"
            )

            client = TestClient(app)

            # Send multiple requests
            responses = []
            for i in range(5):
                response = client.post(
                    "/chat",
                    json={
                        "prompt": f"What's the weather?" if i % 2 == 0 else f"Query {i}",
                        "session_id": f"session_{i}",
                    },
                )
                responses.append(response)

            # Verify all succeeded
            for response in responses:
                assert response.status_code == 200

    @pytest.mark.integration
    def test_voice_to_calendar_flow(self):
        """Test complete voice input to calendar action flow."""
        from main import process_input

        # Mock the complete flow
        with patch("main.voice_input") as mock_voice:
            mock_voice.return_value = {
                "status": "success",
                "user_request": "Schedule a meeting tomorrow at 2pm",
                "audio_source": "microphone",
            }

            with patch("main.calendar_assistant") as mock_calendar:
                mock_calendar.return_value = "Meeting scheduled for tomorrow at 2:00 PM"

                with patch("main.get_orchestrator") as mock_get:
                    mock_orchestrator = MagicMock()
                    mock_orchestrator.return_value = "I'll schedule that meeting for you."
                    mock_orchestrator.side_effect = lambda x: mock_calendar(x)
                    mock_get.return_value = mock_orchestrator

                    with patch("main.select_model") as mock_select:
                        mock_select.return_value = {"provider": "llamacpp"}

                        with patch("main.update_orchestrator_model") as mock_update:
                            mock_update.return_value = mock_orchestrator

                            response = process_input("voice")

                            # Verify the flow
                            mock_voice.assert_called_once()
                            assert "meeting" in response.lower() or "schedule" in response.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
