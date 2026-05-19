# Phase 3: Refactoring Rules & Cleanup

## 1. The Cleanup (Destructive Action)
You MUST permanently delete the following files from the workspace. Do not move them; delete them:
- `lumos_server_temp.py`
- `nova_network_models_temp.py`

## 2. Updating the Imports (CRITICAL)
Because the files have moved into a `src/` hierarchy, all Python import statements across the entire project must be updated to use absolute imports based on the new structure.
- **Example Old:** `from nova_network_models import BaseEvent`
- **Example New:** `from src.network.nova_network_models import BaseEvent`
- **Example Old:** `from nova_yolo_worker import NovaYoloWorker`
- **Example New:** `from src.workers.nova_yolo_worker import NovaYoloWorker`

## 3. Pathing Fixes
- In `src/network/nova_server.py`, ensure the path to `nova_brain.pkl` points to `data/face_db/nova_brain.pkl`. (Create the `data/face_db/` directories if they do not exist).
- In `config/nova_config_manager.py`, ensure `CONFIG_PATH` correctly resolves to the new `config/hazards_config.json` location relative to the project root.

## 4. Execution Constraint
- Do NOT alter any internal business logic, threading, or computer vision code during this step. Your ONLY job is to move files, delete the temps, and fix the import paths so the project runs exactly as it did before, just from a cleaner structure.