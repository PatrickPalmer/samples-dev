# Agentic AI at the Edge - Demo Guide

## Overview

This guide provides a practical, step-by-step demonstration of the Agentic AI system powered by **Qwen3-1.7B** from ![Hugging Face](https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-pirate.png) **Hugging Face** across development, container, and edge deployments. The system features advanced reasoning capabilities with thinking/non-thinking mode switching and FFmpeg-Whisper integration for superior voice processing.

## Prerequisites

### System Requirements
- **Development Machine**: macOS, Linux, or Windows with Python 3.8+
- **Docker**: Version 20.10+ installed and running
- **Memory**: Minimum 8GB RAM (12GB recommended for Qwen3-1.7B)
- **Storage**: 5GB free space for Qwen3 model and Whisper

### Pre-Demo Checklist
```bash
# 1. Clone and setup repository
git clone <repository>
cd agentic-ai-at-the-edge

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models (one-time setup)
mkdir -p models && cd models
huggingface-cli download Qwen/Qwen3-1.7B-Instruct-GGUF qwen3-1.7b-instruct-q8_0.gguf --local-dir .
huggingface-cli download ggerganov/whisper.cpp ggml-base.bin --local-dir .
cd ..

# 4. Start llama-server
llama-server -m models/qwen3-1.7b-instruct-q8_0.gguf \
  --host 0.0.0.0 --port 8080 -c 32768 --chat-template qwen3

# 5. Verify vehicle knowledge base exists
ls src/vehicle_knowledge/data/
# Should see: documents.json, embeddings.npy, vehicle.index (TechCar Model X data)

# 6. Verify FFmpeg with Whisper support
ffmpeg -hide_banner -filters 2>&1 | grep whisper
# Should show: whisper ... Transcribe audio using whisper.cpp
```

## Audio Support Across All Modes

The application now supports voice input seamlessly across all deployment modes:

1. **Development Mode**: Direct microphone access (works natively)
2. **Container Mode**: Audio file exchange via volume mount
   - Host records audio to `audio_exchange/voice_input.wav`
   - Container detects and processes the file
3. **API Mode**: Base64-encoded audio in JSON payload
   - Client records audio and encodes to base64
   - Sends via POST request with `audio_data` field

### Audio CLI Usage
```bash
# For container mode - record audio to shared volume
python -m src.utils.audio_cli record --duration 10

# For API mode - record and send to API
python -m src.utils.audio_cli api --duration 10 --url http://localhost:8000/chat
```

## Demo Flow

### Part 1: Cockpit Control Demonstration (8 minutes) 

**Goal**: Showcase direct vehicle control capabilities matching automotive requirements

#### Setup
```bash
# Terminal 1: Keep llama-server running

# Terminal 2: Start with vehicle controls
python main.py
```

#### Demo Script - Cockpit Controls

**1. Climate Control**
```
USER: Set the temperature to 72 degrees

A: ✅ Climate set to 72°F
Current cabin temperature: 68°F
Fan speed: 3/7
Mode: auto
AC: On

USER: I'm too hot

A: ✅ Climate set to 69°F
Current cabin temperature: 72°F
Fan speed: 3/7
Mode: auto
AC: On

USER: Turn on max AC

A: ✅ Maximum cooling activated
Temperature set to 60°F (minimum)
Fan speed set to maximum (7/7)
AC compressor on
Recirculation enabled
```

**2. Window Control**
```
USER: Open the driver window

A: ✅ Driver window opened to 100%
CAN signal: 0x3B2 [01 64 00 00] (Vehicle in motion at 0 mph) - 3.0 seconds

USER: Close all windows

A: ✅ All windows closed
• Driver window: closed
• Passenger window: closed
• Rear left window: closed
• Rear right window: closed
CAN signals: 0x3B2 [00 00 00 00], 0x3B3 [00 00 00 00]

USER: It's stuffy in here

A: ✅ All windows vented to 15% for fresh air
• Driver window: 15% open
• Passenger window: 15% open
• Rear windows: 15% open
CAN signals: 0x3B2 [01 0F 0F 0F]
```

**3. Seat Control**
```
USER: Move my seat back

A: ✅ Driver seat moved back
Position now at 40% forward
CAN signal: 0x2A1 [00 28 32 00] (5 seconds)

USER: Turn on seat heating

A: ✅ Driver seat heating set to level 2/3
Heating will reach target in about 2 minutes
CAN signal: 0x2A3 [02 00]

USER: Save this as memory position 1

A: ✅ Driver seat position saved to memory 1
• Position: 40% forward, 50% height
• Recline: 30°
• Lumbar: Level 3/5
CAN signal: 0x2A5 [01 28 32 1E 03]
```

**4. Lighting Control**
```
USER: Turn on the headlights

A: ✅ Headlights turned on
Mode: on
CAN signal: 0x1F4 [02]

USER: Set ambient lighting to blue

A: ✅ Ambient lighting set to blue at 60% intensity
CAN signal: 0x4D2 [03 3C]

USER: Turn on my reading light

A: ✅ Driver reading light turned on
CAN signal: 0x4D3 [01 01]
```

**5. Drive Mode Control**
```
USER: Switch to sport mode

A: Sport mode engaged:
- Enhanced throttle response for quicker acceleration
- Firmer steering feel for better feedback
- Tighter suspension (AutoDrive CX-7)
- Engine sound enhancement active
- Fuel economy reduced by ~15%
CAN signal: 0x3E8 [01]

USER: Enable eco mode

A: Eco mode engaged:
- Optimized throttle for fuel efficiency
- Earlier upshifts for lower RPM
- Climate control optimization
- Regenerative braking enhanced
- Fuel economy improved by ~10-15%
CAN signal: 0x3E8 [02]
```

**Key Points to Highlight**:
- Direct vehicle control via natural language
- Safety interlocks (no seat adjustment while driving)
- CAN bus signal generation for authenticity
- Virtual ECU maintains complete vehicle state
- All controls work offline on edge device

### Part 2: Development Mode with Personalization (6 minutes)

**Goal**: Show interactive AI assistant with personalized responses for different drivers

#### Setup
```bash
# Terminal 1: Keep llama-server running

# Terminal 2: Start as Sarah (AutoDrive CX-7 driver)
DRIVER_PROFILE=sarah python main.py

# Or Terminal 3: Start as Bob (Tesla Model 3 driver)
DRIVER_PROFILE=bob python main.py

# Or run without profile for generic mode
python main.py
```

#### Demo Script

**1. Personalized Introduction (NEW)**
```
# Running as Sarah
=============================================================================
              AGENTIC AI AT THE EDGE
=============================================================================
Deployment: Development | Memory: 8g | Context: 100
Driver: Sarah | Vehicle: AutoDrive CX-7 2023

USER: Hello

A: Hello Sarah! I'm here to help with your AutoDrive CX-7. How can I assist you today?

USER: What's my tire pressure?

A: Hi Sarah! For your AutoDrive CX-7, the recommended tire pressure is:
- Front tires: 35 PSI (2.4 bar)  
- Rear tires: 42 PSI (2.9 bar)
Remember to check when tires are cold for accurate readings. Your CX-7's run-flat tires can handle temporary pressure loss safely.
```

**Key Points to Highlight**:
- Personalized greeting with driver name
- Vehicle-specific information (AutoDrive CX-7 vs generic)
- Communication style matches profile (detailed for Sarah)

**2. Contrast with Bob's Profile**
```
# Switch to Bob's profile (Tesla Model 3)
# Ctrl+C and restart with:
DRIVER_PROFILE=bob python main.py

USER: What's my battery status?

A: Bob, your Model 3 battery: 78% charge, 312 miles range.

USER: How do I enable autopilot?

A: Double-tap down on the gear stalk. Requires clear lane markings.
```

**Key Points**:
- Brief, data-focused responses for Bob
- Tesla-specific features (battery, autopilot)
- No extra explanations (Bob is tech-savvy)

**3. Voice Input Works with Personalization**

USER: voice
[SPEAK for 10 seconds]: "Schedule a team meeting tomorrow at 2 PM to discuss the quarterly sensor integration project"

A: I'll schedule that meeting for you. Created appointment:
- Title: Team Meeting - Quarterly Sensor Integration
- Date: [Tomorrow's date] 
- Time: 2:00 PM
- ID: apt_001

USER: What's on my agenda for tomorrow?

A: Your agenda for tomorrow:
- 2:00 PM: Team Meeting - Quarterly Sensor Integration (apt_001)
```

**Key Points to Highlight**:
- Voice transcription using FFmpeg-Whisper for optimal accuracy
- Qwen3-1.7B advanced reasoning with thinking mode capabilities
- Natural language understanding with 32K context window
- Calendar agent automatically invoked with intelligent scheduling
- Personalization context maintained across conversations

**4. Vehicle Assistant with Personalization**
```
# As Sarah (AutoDrive CX-7)
USER: My tire pressure warning light just came on. What should I do?

A: Sarah, I understand this can be concerning! For your AutoDrive CX-7's TPMS warning:

**Immediate Steps:**
1. Your run-flat tires allow safe driving up to 50 miles at reduced speed
2. Check all tires visually for damage
3. Visit your AutoDrive dealer or any tire shop to check pressure

- Front tires: 35 PSI (cold)
- Rear tires: 42 PSI (cold)
- Run-flat capability: 50 miles at 50 mph max

**AutoDrive TPMS Reset:**
1. Use infotainment menu → Vehicle Settings → Tire Pressure → Reset
2. Confirm the reset
3. Drive for a few minutes to complete calibration

I've also scheduled a reminder for you to check this tomorrow. Stay safe!

# Compare with generic response (no profile):
USER: My tire pressure warning light just came on. What should I do?

A: I found information about TPMS (Tire Pressure Monitoring System):
[Generic response without personalization...]
```

**Key Points**:
- Offline vehicle knowledge (FAISS vector search)
- Practical automotive guidance
- No internet required

**3. Dynamic Model Selection**
```
USER: What's 2+2?

A: [Local model selected for simple query]
4

USER: Write a comprehensive marketing strategy for electric vehicle charging stations targeting urban millennials

A: [Cloud model selected for complex query - if available]
[Detailed marketing strategy response...]
```

**Key Points**:
- Simple queries use fast local model
- Complex queries route to cloud (when available)
- Automatic selection based on query analysis

### Part 2: Container Deployment (5 minutes)

**Goal**: Show same functionality in containerized environment

#### Setup
```bash
# Terminal 3: Build and run container
cd src/edge/deployment
docker build -f Dockerfile.edge -t agentic-ai-edge ../../../

# Run in interactive mode
docker run -it --rm \
  -p 8080:8080 \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/audio_exchange:/app/audio_exchange \
  --name edge-assistant \
  agentic-ai-edge

# Terminal 4: Connect to container
docker exec -it edge-assistant python /app/main.py
```

#### Demo Script

**1. Container with Personalization**
```bash
# Run container with Sarah's profile
docker run -it --rm \
  -p 8080:8080 \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/audio_exchange:/app/audio_exchange \
  -e DRIVER_PROFILE=sarah \
  --name edge-assistant \
  agentic-ai-edge
```

**2. Container Health Check**
```
USER: Tell me about your current environment

A: I'm running in a containerized edge environment:
- Deployment: edge
- Model: Local LlamaCpp
- Memory Limit: 8GB
- Context Window: 8192 tokens
- All processing happens locally for privacy

USER: voice

# Terminal 4 (on host Mac): Use the audio CLI
python -m src.utils.audio_cli record --duration 10
[SPEAK]: "What maintenance is due on my TechCar Model X with 25,000 miles?"
# Audio saved to audio_exchange/voice_input.wav

# Back in container terminal - it detects the file automatically
A: [Processing voice input from file...]
Voice captured: "What maintenance is due on my TechCar Model X with 25,000 miles?"

A: Based on the TechCar Model X maintenance schedule at 25,000 miles:

**Due Now:**
- Oil change with full synthetic 0W-20 (every 10,000 miles)
- Tire rotation (every 5,000 miles)
- Multi-point inspection

**Service Details:**
- Oil capacity: 5.5 quarts including filter
- Use manufacturer-approved oil filters only
- Cabin air filter check (replace at 30,000 miles)

**Upcoming at 30,000 miles:**
- Cabin air filter replacement (Part: TCX-CAF-2024)
- Battery test (12V battery typically lasts 4-6 years)
- Brake pad inspection (fronts last 40,000-60,000 miles)

**Your TechCar Model X Features:**
- 2.0L turbocharged engine (250 hp)
- 8-speed automatic transmission
- Full synthetic oil for extended intervals

Estimated cost: $120-150 for current service
```

**Key Points**:
- Container provides identical functionality
- Resource constraints handled automatically
- Voice input works in container with audio device access
- Personalization works through environment variables

**3. Show Container Resource Usage**
```bash
# Terminal 5: Monitor resources
docker stats edge-assistant

# Should show:
# CPU: 15-25% during inference
# Memory: 3-4GB usage
# Efficient resource utilization
```

### Part 3: Edge API Service Mode (5 minutes)

**Goal**: Demonstrate automotive integration via REST API

#### Setup
```bash
# Terminal 6: Run container in API mode
docker run -d --rm \
  -p 8000:8000 \
  -e ENABLE_API=true \
  -v $(pwd)/models:/app/models \
  --name edge-api \
  agentic-ai-edge

# Verify API is running
curl http://localhost:8000/health
```

#### Demo Script

**1. API Health & Status**
```bash
curl http://localhost:8000/health

# Response:
{
  "status": "healthy",
  "deployment": "automotive",
  "model": "llamacpp",
  "capabilities": ["chat", "voice", "image", "offline"]
}
```

**2. API with Personalization**
```bash
# Note: API mode doesn't support DRIVER_PROFILE env var yet,
# but session_id can be used to track different users
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, what car am I driving?",
    "session_id": "sarah_session"
  }'

# For now returns generic TechCar Model X response
# Future enhancement: Map session_id to driver profiles
```

**3. Voice Input via API**
```bash
# Use the audio CLI to record and send voice to API
python -m src.utils.audio_cli api --duration 10 --session-id driver_001
[SPEAK]: "The check engine light just came on in my TechCar Model X. Is it safe to continue?"

# The audio is automatically recorded, encoded, and sent to the API
# Response:
{
  "response": "**TechCar Model X - Check Engine Light Assessment**\n\nBased on your vehicle's manual, the check engine light (yellow engine symbol) indicates:\n\n✅ **SAFE to continue** if:\n- Light is solid yellow (not flashing)\n- No other red warning lights\n- Engine running smoothly\n- Temperature gauge normal\n\n⚠️ **STOP immediately if**:\n- Red oil pressure warning\n- Red battery symbol\n- Red temperature warning\n- Flashing check engine light\n\n**TechCar Model X Specifics:**\n- Your 2.0L turbo engine has enhanced diagnostics\n- Common causes: emissions sensor, turbo wastegate\n- Safe to drive to dealer within 50 miles\n\n**Next Steps:**\n1. Check gas cap is tight (common cause)\n2. Note any performance changes\n3. Schedule diagnostic scan\n\nIs the light solid or flashing?",
  "session_id": "driver_001"
}
```

**3. Session Context Demonstration**
```bash
# Follow-up maintains context
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "No other symptoms, but where can I get the diagnostic scan?",
    "session_id": "driver_001"
  }'

# Response shows context awareness:
{
  "response": "Good - with no other symptoms, you can safely drive to get diagnosed.\n\n**Free OBD-II Diagnostic Scans:**\n\n🔧 **Auto Parts Stores** (Free):\n- AutoZone\n- O'Reilly Auto Parts  \n- Advance Auto Parts\n- NAPA (some locations)\n\n**What They Provide:**\n- Read error codes\n- Basic explanation\n- Reset option (if safe)\n- Printed report\n\n**Alternative Options:**\n- Many oil change shops include free scans\n- Your regular mechanic\n- Buy OBD-II reader ($20-50) for future use\n\n**When You Go:**\n- Bring your registration\n- Takes 5-10 minutes\n- Write down the code (P0xxx format)\n- Ask if it's safe to clear\n\nWant me to explain what common codes mean?",
  "session_id": "driver_001"
}
```

**4. Multi-Modal API Demo**
```bash
# Schedule service via API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Schedule an oil change for next Monday at 9 AM",
    "session_id": "driver_001"
  }'

# Response:
{
  "response": "✅ **Oil Change Scheduled**\n\n📅 **Appointment Details:**\n- Service: Oil Change\n- Date: Monday [next Monday's date]\n- Time: 9:00 AM\n- Duration: 30-45 minutes\n- ID: svc_oil_001\n\n📋 **Preparation:**\n- Current mileage for records\n- Preferred oil type (synthetic/conventional)\n- Any other concerns to address\n\n💡 **While You're There:**\n- Tire rotation (if due)\n- Air filter check\n- Fluid top-offs (usually free)\n\nNeed directions or want to add services?",
  "session_id": "driver_001"
}
```

**Key Points**:
- RESTful API perfect for automotive systems
- Session management for conversation context
- Structured responses for easy parsing
- All processing happens locally (privacy)

### Part 4: Performance & Architecture Benefits (3 minutes)

**Goal**: Highlight technical achievements with Qwen3-1.7B and FFmpeg-Whisper

#### 1. Show Unified Codebase
```bash
# Same main.py runs everywhere
ls src/edge/edge_agent.py 2>/dev/null || echo "No edge_agent.py - unified codebase!"

# Show deployment awareness
grep -n "DEPLOYMENT_TARGET" src/config.py
grep -n "ENABLE_API" main.py
```

#### 2. Demonstrate Model Selection
```
# In development mode
USER: Write a comprehensive analysis of quantum computing applications in automotive sensors

A: [Qwen3 thinking mode activated for complex reasoning]
<think>This is a complex technical question requiring deep analysis of quantum computing principles and their potential applications in automotive sensor technology...</think>
[Detailed technical analysis with superior reasoning capabilities...]

# In edge/container mode
USER: Write a comprehensive analysis of quantum computing applications in automotive sensors

A: [Local model handles the query]
[Response generated using local LlamaCpp model...]
```

## Common Issues & Solutions

### Model Download Issues
```bash
# If Hugging Face is slow:
wget https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF/resolve/main/Qwen2.5-Omni-7B-Q4_K_M.gguf
```

### Voice Input Not Working
```bash
# Check audio devices
docker run --device /dev/snd ...

# On macOS, may need:
brew install portaudio
```

### API Connection Refused
```bash
# Ensure port mapping
docker run -p 8000:8000 ...

# Check container logs
docker logs edge-api
```

## Summary

This demo showcases a production-ready automotive AI assistant that:

**Core Capabilities:**
- **Advanced AI Reasoning**: Qwen3-1.7B with thinking/non-thinking mode switching
- **Cockpit Control**: Direct control of climate, windows, seats, lights, and drive modes
- **Vehicle Knowledge**: Comprehensive automotive documentation with multilingual support
- **Superior Voice Processing**: FFmpeg-Whisper integration for optimal transcription
- **Personalization**: Driver profiles with intelligent, context-aware responses
- **Edge Deployment**: Runs the same code everywhere (dev → container → edge)
- **Privacy First**: All processing happens locally, no cloud dependency
- **Open Source**: Apache 2.0 licensed Qwen3 from Hugging Face

**Technical Achievements:**
- **Advanced AI Integration**: Qwen3-1.7B with thinking mode for complex reasoning
- **FFmpeg-Whisper Pipeline**: Optimized voice transcription with native integration
- Virtual ECU with complete vehicle state management
- CAN bus signal generation for authentic automotive integration
- Safety interlocks with intelligent context awareness
- Advanced natural language understanding with 32K context window
- Offline FAISS vector search with multilingual vehicle knowledge
- Dynamic model selection (Qwen3 local vs cloud when available)
- REST API for automotive system integration
- Efficient operation (2-4GB RAM for 1.7B parameter model)
- **Open Source Foundation**: Apache 2.0 licensed components from Hugging Face

**Demonstration Highlights:**
1. Cockpit controls with safety validation (8 min)
2. Personalized driver profiles (6 min)
3. Container deployment (5 min)
4. API integration (5 min)
5. Performance & architecture (3 min)

Total demo time: 22-25 minutes
Setup time: 10-15 minutes (one-time)