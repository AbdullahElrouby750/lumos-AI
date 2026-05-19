# Phase 3: The Great Refactor (Project Restructure)
**Objective:** Reorganize the flat root directory into a production-grade Python package structure. Update all inter-module imports to reflect the new hierarchy.

## 1. The Target Directory Structure
Create these folders and move the corresponding files into them:

* **`/config/`**
  - `hazards_config.json`
  - `nova_config_manager.py`
* **`/models/`**
  - `yolov11n.pt` (if exists)
  - `face_landmarker.task`
  - `blaze_face_short_range.tflite`
* **`/docs/`**
  - Move ALL `.md` files here EXCEPT `README.md`.
* **`/local_tests/`** (For laptop debugging & legacy reference)
  - `test.py`
  - `yoloTest.py`
  - `mock_voice_client.py`
  - `vision.py` (The teammate's original script - archive it here)
* **`/src/core/`**
  - `nova_vision_pipeline.py`
  - `nova_commands.py`
  - `nova_audio.py`
  - `utils.py`
  - `keys.py`
* **`/src/network/`**
  - `nova_server.py`
  - `nova_discovery.py`
  - `nova_network_models.py`
* **`/src/workers/`**
  - `nova_yolo_worker.py`
  - `nova_ai_worker.py`
* **`/src/modules/`**
  - `brain_module.py`
  - `OCR.py`
  - `ORS.py`
  - `nova_enrollment.py`
  - `nova_face_detector.py`
  - `nove_face_encoder.py`
  - `nove_face_rec.py`
  - `nove_forget.py`
  - `nova_listener.py`
  - `nova_pose_validator.py`

## 2. Root Directory (Keep These Here)
- `main.py`
- `requirements.txt`
- `README.md`
- `.gitignore`