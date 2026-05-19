# Phase 2 (Part 2): On-Demand AI Workers (Gemini & OCR)
**Objective:** Integrate the teammate's Gemini and EasyOCR logic into the Nova architecture as thread-safe, on-demand workers that never block the camera loop.

## 1. The On-Demand Architecture
- Create `nova_ai_worker.py`.
- This worker will **not** consume frames continuously. It runs a daemon thread that waits on a `queue.Queue[dict]` for explicit user commands.
- **Command Structure:** `{"intent": "INTENT_SCENE"}` or `{"intent": "INTENT_TEXT"}`.

## 2. The RAM-Disk Bridge (The Eyes)
- When a command is received, the worker will read the most recent high-resolution frame directly from the Pi's RAM disk (`/dev/shm/latest_scene.jpg`).

## 3. The Teammate Wrapper (The Brain)
- Import the processing functions from `brain_module.py` and `OCR.py`.
- Pass the RAM-disk image to these functions.
- **Crucial:** Wrap all calls to Gemini and OCR in strict `try/except` blocks to handle network timeouts or API failures without crashing the thread.

## 4. Unified Output (The Mouth)
- Send the resulting text back to the Flutter app using `self.server.send_event()`.
- Use `OCRResultEvent` for text reading (Priority: `PRIORITY_INFO`).
- Use standard `BaseEvent` for scene descriptions (Priority: `PRIORITY_LOW`).