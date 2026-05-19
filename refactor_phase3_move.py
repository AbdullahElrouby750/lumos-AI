import os
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent
new_dirs = [
    root / 'models',
    root / 'docs',
    root / 'local_tests',
    root / 'src' / 'core',
    root / 'src' / 'network',
    root / 'src' / 'workers',
    root / 'src' / 'modules',
    root / 'data' / 'face_db',
]
for d in new_dirs:
    d.mkdir(parents=True, exist_ok=True)

moves = {
    root / 'nova_config_manager.py': root / 'config' / 'nova_config_manager.py',
    root / 'hazards.py': root / 'src' / 'modules' / 'hazards.py',
    root / 'face_landmarker.task': root / 'models' / 'face_landmarker.task',
    root / 'blaze_face_short_range.tflite': root / 'models' / 'blaze_face_short_range.tflite',
    root / 'yolov8n.pt': root / 'models' / 'yolov8n.pt',
    root / 'test.py': root / 'local_tests' / 'test.py',
    root / 'yoloTest.py': root / 'local_tests' / 'yoloTest.py',
    root / 'mock_voice_client.py': root / 'local_tests' / 'mock_voice_client.py',
    root / 'vision.py': root / 'local_tests' / 'vision.py',
    root / 'nova_vision_pipeline.py': root / 'src' / 'core' / 'nova_vision_pipeline.py',
    root / 'nova_commands.py': root / 'src' / 'core' / 'nova_commands.py',
    root / 'nova_audio.py': root / 'src' / 'core' / 'nova_audio.py',
    root / 'utils.py': root / 'src' / 'core' / 'utils.py',
    root / 'keys.py': root / 'src' / 'core' / 'keys.py',
    root / 'nova_server.py': root / 'src' / 'network' / 'nova_server.py',
    root / 'nova_discovery.py': root / 'src' / 'network' / 'nova_discovery.py',
    root / 'nova_network_models.py': root / 'src' / 'network' / 'nova_network_models.py',
    root / 'nova_yolo_worker.py': root / 'src' / 'workers' / 'nova_yolo_worker.py',
    root / 'nova_ai_worker.py': root / 'src' / 'workers' / 'nova_ai_worker.py',
    root / 'brain_module.py': root / 'src' / 'modules' / 'brain_module.py',
    root / 'OCR.py': root / 'src' / 'modules' / 'OCR.py',
    root / 'ORS.py': root / 'src' / 'modules' / 'ORS.py',
    root / 'nova_enrollment.py': root / 'src' / 'modules' / 'nova_enrollment.py',
    root / 'nova_face_detector.py': root / 'src' / 'modules' / 'nova_face_detector.py',
    root / 'nove_face_encoder.py': root / 'src' / 'modules' / 'nove_face_encoder.py',
    root / 'nove_face_rec.py': root / 'src' / 'modules' / 'nove_face_rec.py',
    root / 'nove_forget.py': root / 'src' / 'modules' / 'nove_forget.py',
    root / 'nova_listener.py': root / 'src' / 'modules' / 'nova_listener.py',
    root / 'nova_pose_validator.py': root / 'src' / 'modules' / 'nova_pose_validator.py',
}
for src, dst in moves.items():
    if src.exists():
        print(f"Moving {src.name} -> {dst}")
        shutil.move(str(src), str(dst))

for md in root.glob('*.md'):
    if md.name != 'README.md':
        print(f"Moving doc {md.name} -> docs")
        shutil.move(str(md), str(root / 'docs' / md.name))

for temp in [root / 'lumos_server_temp.py', root / 'nova_network_models_temp.py']:
    if temp.exists():
        print(f"Deleting {temp.name}")
        temp.unlink()

brain_src = root / 'nova_brain.pkl'
brain_dst = root / 'data' / 'face_db' / 'nova_brain.pkl'
if brain_src.exists():
    print(f"Moving brain file -> {brain_dst}")
    shutil.move(str(brain_src), str(brain_dst))

for asset in ['bus.jpg']:
    p = root / asset
    if p.exists():
        print(f"Moving asset {p.name} -> local_tests")
        shutil.move(str(p), str(root / 'local_tests' / p.name))

face_db_dir = root / 'face_db'
if face_db_dir.exists() and face_db_dir.is_dir():
    try:
        face_db_dir.rmdir()
        print("Removed empty root face_db folder")
    except OSError:
        print("Root face_db folder not empty, left in place")
