#!/bin/bash
# startup.sh - Edge Startup Script
# Handles model download, llama.cpp server startup, and agent initialization
# 
# For local development with custom FFmpeg+Whisper:
# export FFMPEG_PATH="/path/to/your/ffmpeg-whisper/bin/ffmpeg"
# export WHISPER_MODEL_PATH="/path/to/your/whisper/models/ggml-base.bin"
# export DYLD_LIBRARY_PATH="/path/to/whisper/libs" (macOS only)

set -e

# Configuration from environment (set by docker run from .env)
# Exit if required variables are not set
if [ -z "$MODEL_PATH" ]; then
    echo "❌ MODEL_PATH not set. Container must be run with environment variables from .env"
    exit 1
fi

# Use environment variables (no defaults - force .env usage)
MODEL_PATH=${MODEL_PATH}
WHISPER_MODEL_PATH=${WHISPER_MODEL_PATH}
SERVER_HOST=${SERVER_HOST:-"0.0.0.0"}  # Network defaults OK
SERVER_PORT=${SERVER_PORT:-"8080"}
THREADS=${GGML_NTHREADS:-"4"}
GPU_LAYERS=${GPU_LAYERS:-"0"}
FFMPEG_PATH=${FFMPEG_PATH:-"ffmpeg"}
CHAT_TEMPLATE=${CHAT_TEMPLATE}
CONTEXT_SIZE=${LLAMA_CTX_SIZE}

echo "🚀 Starting Personal Assistant Edge Container..."
echo "=================================================="
echo "📁 Model directory: $(dirname $MODEL_PATH)"
echo "🎯 Target model: $(basename $MODEL_PATH)"
echo "🎤 Whisper model: $(basename $WHISPER_MODEL_PATH)"
echo "🌐 Server: ${SERVER_HOST}:${SERVER_PORT}"
echo "=================================================="

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for server to be ready
wait_for_server() {
    local url="http://localhost:${SERVER_PORT}/health"
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for llama.cpp server to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo "✅ Server is ready!"
            return 0
        fi
        
        echo "⏳ Attempt $attempt/$max_attempts - waiting for server..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ Server failed to start within expected time"
    return 1
}

# Function to detect optimal thread count
detect_threads() {
    if command_exists nproc; then
        local cpu_count=$(nproc)
        # Cap at 4 threads for edge devices to prevent resource contention
        echo $((cpu_count > 4 ? 4 : cpu_count))
    else
        echo "2"  # Safe default
    fi
}

# Auto-detect optimal thread count if not set
if [ "$THREADS" = "auto" ]; then
    THREADS=$(detect_threads)
    echo "🔧 Auto-detected $THREADS threads"
fi

# Validate volume mounts and directories
echo ""
echo "🔍 Validating directories and mounts..."

# Check if models directory is writable
if [ ! -w "$(dirname $MODEL_PATH)" ]; then
    echo "⚠️  Warning: Models directory not writable: $(dirname $MODEL_PATH)"
    echo "   Model download may fail"
fi

# Check for audio exchange directory (for macOS Docker)
if [ -d "/app/audio_exchange" ]; then
    echo "✅ Audio exchange directory mounted (macOS file-based audio)"
    if [ ! -w "/app/audio_exchange" ]; then
        echo "⚠️  Warning: Audio exchange directory not writable"
        echo "   Voice input may not work properly"
    fi
fi

# Step 1: Check if models exist, download if needed
echo ""
echo "🔍 Checking for required models..."

if [ ! -f "$MODEL_PATH" ]; then
    echo "📥 Model not found, downloading..."
    # Retry download up to 3 times
    for attempt in 1 2 3; do
        echo "Download attempt $attempt of 3..."
        python /app/model_downloader.py
        
        if [ $? -eq 0 ]; then
            echo "✅ Model download succeeded"
            break
        else
            echo "⚠️ Download attempt $attempt failed"
            if [ $attempt -eq 3 ]; then
                echo "❌ Model download failed after 3 attempts"
                # Check disk space
                df -h $(dirname $MODEL_PATH)
                # Don't exit - try to continue with any existing models
                echo "⚠️ Continuing with existing models if available..."
            else
                echo "Retrying in 5 seconds..."
                sleep 5
            fi
        fi
    done
else
    echo "✅ Model found: $(basename $MODEL_PATH) ($(du -h "$MODEL_PATH" | cut -f1))"
fi

# Check Whisper model for voice support
if [ ! -f "$WHISPER_MODEL_PATH" ]; then
    echo "⚠️  Whisper model not found: $(basename $WHISPER_MODEL_PATH)"
    echo "   Voice transcription will not be available"
else
    echo "✅ Whisper model found: $(basename $WHISPER_MODEL_PATH) ($(du -h "$WHISPER_MODEL_PATH" | cut -f1))"
fi

# Check FFmpeg Whisper support
echo "🔍 Checking FFmpeg Whisper support..."
if command -v "$FFMPEG_PATH" &> /dev/null; then
    if "$FFMPEG_PATH" -hide_banner -filters 2>&1 | grep -q "whisper.*Transcribe audio"; then
        echo "✅ FFmpeg with Whisper filter found: $FFMPEG_PATH"
    else
        echo "⚠️  FFmpeg found but without Whisper filter: $FFMPEG_PATH"
        echo "   Voice transcription may not work properly"
    fi
else
    echo "⚠️  FFmpeg not found at: $FFMPEG_PATH"
    echo "   Voice transcription will not be available"
fi

# Step 2: Start llama.cpp server
echo ""
echo "🚀 Starting llama.cpp server..."
echo "   Model: $(basename $MODEL_PATH)"
echo "   Host: $SERVER_HOST"
echo "   Port: $SERVER_PORT"
echo "   Context: $CONTEXT_SIZE"
echo "   Threads: $THREADS"
echo "   GPU Layers: $GPU_LAYERS"

# Check if llama-server binary exists
if [ ! -f "/usr/local/bin/llama-server" ]; then
    echo "❌ llama-server binary not found at /usr/local/bin/llama-server"
    echo "   Checking other locations..."
    which llama-server || echo "   llama-server not found in PATH"
    exit 1
fi

echo "🔧 Starting llama.cpp server with native binary..."

# Start llama-server with optimized configuration
echo "🔧 Starting llama-server with $CHAT_TEMPLATE template..."
echo "   Context: $CONTEXT_SIZE, Threads: $THREADS, Batch: ${LLAMA_BATCH_SIZE:-512}"

# Start llama.cpp server with optimizations
/usr/local/bin/llama-server \
    --model "$MODEL_PATH" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    -c "$CONTEXT_SIZE" \
    --threads "$THREADS" \
    --threads-batch "$THREADS" \
    -ngl "$GPU_LAYERS" \
    --jinja \
    --chat-template "$CHAT_TEMPLATE" \
    --batch-size "${LLAMA_BATCH_SIZE:-512}" \
    --ubatch-size "${LLAMA_UBATCH_SIZE:-512}" \
    --parallel "${LLAMA_N_PARALLEL:-1}" \
    --threads-http "${LLAMA_THREADS_HTTP:-8}" \
    --verbose &

SERVER_PID=$!

echo "🔧 Server started with PID: $SERVER_PID"

# Step 3: Wait for server to be ready
if ! wait_for_server; then
    echo "❌ Server startup failed"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Step 4: Start the personal assistant agent
echo ""
echo "🤖 Starting Personal Assistant Agent..."
echo "=================================================="

# Check if API mode is enabled
ENABLE_API=${ENABLE_API:-"false"}

# Set up signal handlers to gracefully shutdown
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    echo "   Stopping llama.cpp server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "✅ Shutdown complete"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Container should ONLY serve the llama-server API
# main.py will be run locally on the host machine
echo "🎉 Llama.cpp server is ready!"
echo "   Server PID: $SERVER_PID"
echo "   API endpoint: http://localhost:8080"
echo ""
echo "💡 The assistant supports:"
echo "   🎤 Voice input with FFmpeg Whisper"
echo "   🎛️ Cockpit controls (climate, windows, seats, lights, drive modes)"
echo "   🧠 Qwen3 1.7B for efficient edge inference"
echo "   🔧 Tool calling via --jinja flag"
echo ""

# Show volume mount status
echo "📦 Volume mounts:"
ls -la /app/models 2>/dev/null && echo "   ✅ Models directory mounted" || echo "   ⚠️  Models directory not accessible"
ls -la /app/data 2>/dev/null && echo "   ✅ Data directory mounted" || echo "   ⚠️  Data directory not accessible"
ls -la /app/logs 2>/dev/null && echo "   ✅ Logs directory mounted" || echo "   ⚠️  Logs directory not accessible"
[ -d "/app/audio_exchange" ] && echo "   ✅ Audio exchange directory mounted" || echo "   ℹ️  Audio exchange not mounted (Linux audio or no voice)"

echo ""
echo "📊 Resource usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null || echo "   Run 'docker stats' to monitor resource usage"

# Keep container running (just serve llama-server)
wait $SERVER_PID

# If server exits, cleanup
cleanup