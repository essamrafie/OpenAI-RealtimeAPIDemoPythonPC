## OpenAI Realtime Voice Assistant (Python)

Realtime voice assistant that streams microphone audio to OpenAI’s Realtime API and plays back the AI’s audio replies. Includes function-calling tools (weather, notepad, camera vision, robot wave, opening speech), stability hardening, and auto-reconnect.

### How it works

- Microphone audio is captured via `PyAudio` and queued; a sender thread base64-encodes chunks and sends `input_audio_buffer.append` frames over WebSocket.
- The receiver thread processes server events:
  - `session.created` → `send_fc_session_update()` sets instructions, voice, VAD, formats, and tools.
  - `response.audio.delta` → decoded PCM is appended to a playback buffer; a `PyAudio` speaker callback outputs audio in realtime.
  - `response.function_call_arguments.done` → `handle_function_call()` executes local tools and returns results via `send_function_call_result()`.
- Tools available: `get_weather`, `write_notepad` (Windows Notepad), `describe_camera_view` (vision with `gpt-4o-mini`), `robot_wave_hello` (simulated), `opening_ceremony_speech`.
- Stability: IPv4 WS connect with timeouts, heartbeat pings, clean shutdown, and reconnect loop.

### Prerequisites

- Python 3.9+ (Windows 10/11, macOS, or Linux)
- Microphone and speakers
- OpenAI API key with Realtime access
- On Windows: optional camera and Notepad; on Linux/macOS: camera only

### Setup

1) Clone and enter the project
```bash
git clone https://github.com/essamrafie/OpenAI-RealtimeAPIDemoPythonPC.git
cd OpenAI-RealtimeAPIDemoPythonPC
```

2) Create and activate venv
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3) Install dependencies
```bash
pip install -r requirements.txt
```

4) Configure environment
```bash
cp env_template.txt .env  # or create .env
echo OPENAI_API_KEY=sk-... >> .env
```

### Run

```bash
python realtime.py
```

You should see logs: microphone active, connected to OpenAI, and session updates. Speak to interact. If you interrupt while it’s talking, it stops playback and listens.

### Function tools (examples)

- “What’s the weather in Dubai?” → triggers `get_weather` (demo data)
- “Open notepad and write this down ...” → `write_notepad` (Windows)
- “What do you see?” → `describe_camera_view` captures a frame and sends to vision model
- “Wave hello” → `robot_wave_hello` logs a waving motion
- “Start the opening ceremony” → `opening_ceremony_speech`

### Troubleshooting

- Mic/device errors: ensure default mic is available and not in use.
- No audio playback: check speakers and that `speaker_callback` is invoked (logs).
- Camera unavailable: the app continues without vision.
- Socket stalls: reconnect logic and heartbeat should recover; check network/firewall.

### Notes

- Vision uses `gpt-4o-mini` for lower cost; adjust in `realtime.py` if needed.
- Notepad tool is Windows-centric; adapt for Linux/macOS by editing `write_notepad`.