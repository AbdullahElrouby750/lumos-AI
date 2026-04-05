# Lumos Refactor Roadmap: Phases 2 to 5

## Phase 2: Core Architecture Refactor
**Goal:** Break the current monolithic scripts into a modular "CEO/Worker" structure.
**Tasks:**
1. Create `utils.py`: Move all pure math and drawing functions here (distance calculation, bounding box drawing).
2. Create `main.py`: This is the new "CEO." It initializes the camera, runs the `while cap.isOpened():` loop, calls MediaPipe, and orchestrates the other modules.
3. Refactor `nova_face_detector.py` and `nove_face_rec.py` to act as callable functions/classes rather than standalone executable scripts.
**Post-Phase Check:** `main.py` should run, open the webcam, draw basic bounding boxes, and maintain a high FPS without crashing.

## Phase 3: The Voice Queue (Audio Manager)
**Goal:** Solve `pyttsx3` driver crashes (`run loop already started`) using a Producer-Consumer threading model.
**Tasks:**
1. Create `nova_audio.py`.
2. Implement a `VoiceQueue` class with a single background worker thread running a loop to consume text strings.
3. Add **Priority Ranking** (1. DANGER, 2. WARNING, 3. SOCIAL, 4. INFO). The queue should sort by priority.
4. Add **Time-to-Live (TTL)**: If a string is older than 2.0 seconds in the queue, discard it silently to avoid speaking outdated information.
**Post-Phase Check:** Spamming `VoiceQueue.speak("test")` 10 times rapidly should result in sequential audio playback without throwing a threading exception.

## Phase 4: The Vision "Spam Filter"
**Goal:** Prevent the AI from constantly repeating names and smooth out the distance calculations.
**Tasks:**
1. **Centroid Tracking (in `utils.py`):** Assign a temporary ID to detected bounding boxes. 
2. **Asynchronous Recognition:** `main.py` should only send a face crop to `nove_face_rec.py` in a background thread if it is a *new* ID.
3. **The Collision Zone:** Define an invisible vertical column taking up the center 40% of the camera frame.
4. **Alert Logic:** Only trigger a voice alert if:
    a) A newly recognized ID enters the frame.
    b) An existing ID moves *into* the center 40% Collision Zone AND its bounding box width is increasing (meaning it is approaching).
**Post-Phase Check:** The AI should announce a recognized person once. If the person stands still or passes on the far left/right, it remains silent.

## Phase 5: Voice Commands & STT (The NLP Engine)
**Goal:** Replace keyboard inputs ('s' to enroll, 'f' to forget) with an isolated voice command listener.
**Tasks:**
1. Create `mock_voice_client.py`. This script should use `speech_recognition` to listen to the laptop mic and return a simulated Flutter JSON payload: `{"wake_word": True, "raw_text": "[spoken words]", "source": "laptop"}`.
2. Build an **Intent Parser** in `main.py` (or a new `nova_commands.py`) that reads the JSON payload.
3. Map NLP keywords:
    - ["delete", "remove", "forget"] -> Trigger `forget_person()` logic.
    - ["add", "enroll", "remember"] -> Trigger enrollment burst logic.
**Post-Phase Check:** Saying "Hey Lumo, delete Adham" should successfully execute the deletion logic without requiring any keyboard input, and without freezing the camera feed.