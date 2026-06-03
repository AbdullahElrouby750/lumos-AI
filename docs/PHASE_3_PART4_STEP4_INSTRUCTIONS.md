# Phase 3, Part 4 (Step 4): Instructions & Constraints

## 1. Import Modifications (`main.py`)
- **DELETE:** `from src.modules.nova_enrollment import EnrollmentManager`
- **ADD:** `from src.workers.nova_enrollment_worker import NovaEnrollmentWorker`

## 2. Boot Sequence Modifications (`main.py`)
- Instantiate the worker right after `ai_worker`:
  `enrollment_worker = NovaEnrollmentWorker(server)`
- Pass it into the CEO:
  `vision_pipeline = VisionPipeline(..., enrollment_worker=enrollment_worker)`
- **DELETE:** The line `enrollment_manager = EnrollmentManager(voice_queue, vision_pipeline=vision_pipeline)`

## 3. Main Loop Purge (`main.py`)
- **DELETE:** The line `enrollment_manager.update(frame, current_time)` from inside the `while cap.isOpened():` loop.

## 4. Graceful Shutdown (`main.py`)
- Inside the `finally:` block, add `enrollment_worker.stop()` alongside the other worker `.stop()` calls.