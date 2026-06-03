# LUMOS SYSTEM SPECIFICATION (V3.1)

### SCENARIO A: Standard Operation (Reflex Mode)
**What Should Happen:**
1. The `main.py` loop captures video at 30 FPS.
2. The `VisionPipeline` (CEO) processes the frame synchronously. It checks its `_command_inbox` for WebSocket commands via `process_inbox()`.
3. The CEO feeds a deep copy of the frame to the `NovaYoloWorker`.
4. The CEO runs MediaPipe Face Detection. If faces are found, the `CentroidTracker` assigns them temporary IDs.
5. **The Spawn Gate:** If a tracked face is stable for 10 consecutive frames, the CEO spawns a background thread to run DeepFace recognition. It must strictly enforce a ceiling of `_MAX_RECOGNITION_THREADS` using a BoundedSemaphore.
6. **Thread Safety:** The background thread writes the recognized name to `self.recognition_results`. Because the main thread also reads this dictionary, all reads/writes to this dict must be protected by `self._state_lock`.
7. **Audio Output:** If recognized, the CEO sends a text string to the `VoiceQueue`. The `VoiceQueue` packages this into a `BaseEvent` and triggers `server.send_event()` to send the TTS payload over WebSockets to the mobile app.

### SCENARIO B: Spatial / Hazard Alerts
**What Should Happen:**
1. While Scenario A runs, the `VisionPipeline` monitors the bounding boxes of recognized faces. 
2. If a bounding box enters the center 40% of the screen (collision zone) AND the box's area is expanding faster than the historical average (indicating approach), it triggers a `PRIORITY_WARNING` audio event.
3. Simultaneously, the `NovaYoloWorker` is running in its own daemon thread, picking up the copied frames from its queue, and evaluating them for objects/hazards, independent of the main loop.

### SCENARIO C: The Enrollment Protocol (The Badge Office)
**What Should Happen:**
1. The mobile app sends a WebSocket JSON: `{"intent": "INTENT_ENROLL", "target_name": "John"}`.
2. `LumosServer` catches this and triggers the CEO's `on_command_received` callback, safely dropping the JSON into the CEO's `_command_inbox`.
3. Next frame, the CEO reads the inbox, sees `INTENT_ENROLL`, sets `self.is_enrolling = True`, and calls `self.enrollment_worker.start_session("John", on_complete=self._on_enrollment_done)`.
4. **The Optic Nerve Fork:** On the next camera frame, because `is_enrolling` is True, the CEO copies the frame to the `NovaEnrollmentWorker` and *returns early*, completely skipping MediaPipe Face Detection to save CPU and stop "Unknown" spam.
5. **Worker Execution:** The `NovaEnrollmentWorker` processes the frames, validates head poses (straight, left, right, up, down), and saves images to disk. It communicates progress to the phone via `server.send_event()`.
6. **Completion & Reload:** Once 5 poses are captured, the worker runs `build_nova_brain` (DeepFace extraction) in its own thread. When done, it fires the `on_complete` callback. This triggers `hot_reload` in the CEO, which loads the new brain into RAM and resets `is_enrolling = False`, restoring normal camera operation.

### SCENARIO D: High-Level AI Intent (Scene/Text)
**What Should Happen:**
1. The mobile app sends `{"intent": "INTENT_SCENE"}`.
2. The server drops it into the CEO's inbox.
3. The CEO routes the raw command dictionary into the `NovaAIWorker`'s command queue. The AI Worker processes this entirely in the background.

### SCENARIO E: The Forget Protocol
**What Should Happen:**
1. The user asks to delete a face. `nova_commands.py` parses this and outputs `INTENT_FORGET`.
2. The `process_command` function uses fuzzy matching to find the closest name in the database and stores it in the global `pending_forget_name`. It asks the user to confirm via TTS.
3. The mobile app sends `INTENT_CONFIRM`.
4. `process_command` deletes the embeddings from the `.pkl` database and triggers `vision_pipeline.hot_reload()` to wipe the CEO's short-term memory and reload the hard drive database into RAM.

### SCENARIO F: System Shutdown
**What Should Happen:**
1. A hardware interrupt or `INTENT_QUIT` is received.
2. The main `while` loop breaks.
3. The `finally` block in `main.py` executes. It gracefully calls `.stop()` on the `LumosServer`, `NovaYoloWorker`, `NovaAIWorker`, `NovaEnrollmentWorker`, and `VisionPipeline`.
4. All daemon threads exit cleanly without leaking memory or leaving OpenCV windows hanging.