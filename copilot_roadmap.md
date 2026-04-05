# Lumos Final Refactor & Feature Addition

## Task 1: Slicing the "God Object" (`main.py`)
Gut the current `main.py` and move its bloated logic into three new, clean files:
1. **`nova_listener.py`:** Move the `CommandListener` class here. Update the port to `65432`.
2. **`nova_enrollment.py`:** Create an `EnrollmentManager` class. Move all the guided enrollment state machine logic here (counting 5 frames, waiting 3 seconds, queuing voice directions like "look left", and triggering `build_nova_brain`). 
3. **`nova_vision_pipeline.py`:** Create a `VisionPipeline` class. Move the `CentroidTracker` initialization, collision zone math, and the async call to `nove_face_rec.py` here. 

## Task 2: Manual Terminal Overrides (Non-Blocking)
The user needs to manually type names to bypass STT pronunciation errors.
- In the new `main.py` loop, listen for 's' (enroll) and 'f' (forget) in `cv2.waitKey`.
- **Crucial:** Because `input()` blocks the thread, you must spawn a temporary daemon thread when 's' or 'f' is pressed. This thread will run `name = input("Enter name: ")` in the terminal, and then manually push a dictionary (like `{"intent": INTENT_FORGET, "target_name": name}`) into the command queue.

## Task 3: Voice Barge-in (Interruption)
Lumos should shut up when the user starts speaking, unless it's a critical warning.
- In `nova_audio.py`, add a `clear_non_danger_queue()` method to the `VoiceQueue` class. It should empty all pending messages in the priority queue EXCEPT those with `PRIORITY_DANGER` (Priority 1).
- In `nova_listener.py`, the moment a valid JSON payload is received, immediately call `voice_queue.clear_non_danger_queue()` so the AI stops talking and listens.

## Task 4: Voice Command "Quit"
- In `nova_commands.py`, add `INTENT_QUIT` and map it to keywords like ["quit", "exit", "stop", "close"].
- In `main.py`, if `INTENT_QUIT` is pulled from the command queue, break the main camera loop and trigger the graceful shutdown sequence (matching the 'q' key functionality).