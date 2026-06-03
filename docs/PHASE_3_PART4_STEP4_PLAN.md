# Phase 3, Part 4 (Step 4): The Purge
**Objective:** Remove all legacy, blocking enrollment logic from the main application loop and finalize the Dependency Injection of the new `NovaEnrollmentWorker`.

## 1. Legacy Cleanup (`main.py`)
- **Concept:** The `EnrollmentManager` is obsolete. The main camera loop no longer needs to manually feed frames to an enrollment state machine, as the `VisionPipeline` handles optic nerve routing.
- **Execution:** Delete the import and instantiation of `EnrollmentManager`. Delete the `enrollment_manager.update(frame)` call from the 30 FPS camera loop.

## 2. Worker Injection (`main.py`)
- **Concept:** Treat the Badge Office exactly like the AI and YOLO workers.
- **Execution:** - Instantiate `NovaEnrollmentWorker(server)` alongside the other workers.
  - Inject it into the `VisionPipeline`.
  - Add `enrollment_worker.stop()` to the `finally` block to prevent zombie threads.