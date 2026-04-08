Lumos: Social Vision & Face Recognition (Nova Team)
Current Release: Version 3.0 (The "Bulletproof" Production Release)
V3 Milestone: Zero-crash concurrency, Anatomical-Relative Pose Math, and MediaPipe/DeepFace State Synchronization.

👁️ Project Overview
Lumos is a real-time assistive wearable AI for the visually impaired. It transforms raw video into social context, detecting faces, identifying individuals via Deep Learning, and managing spatial proximity alerts without ever dropping below 30 FPS.

The V3 "Bulletproof" Standard
Version 3.0 is engineered to survive the "chaos of the real world." Whether the camera is panning rapidly across a crowded room or being used in a dimly lit hallway, the system is designed to be mathematically stable and thread-safe.

⚙️ V3 Architectural Breakthroughs
1. Zero-Crash Concurrency (Thread-Safe Inference)
The Single-Thread Ceiling: To prevent TensorFlow/DeepFace from "segfaulting" during rapid movement, V3 implements a BoundedSemaphore capped at 1 concurrent inference thread. This ensures the C++ backend never deadlocks.

Safe-Crop Protection: Implements an spatial boundary check that prevents the system from attempting to process "zero-pixel" or negative-coordinate crops when a face is at the edge of the frame.

State Locking: All shared dictionaries are protected by a global _state_lock to prevent Python "Dictionary Tearing" during background updates.

2. Anatomical-Relative Pose Math (T-Zone)
Distance-Independent Validation: The V2 "hard-coded" thresholds were replaced in V3 with Face-Relative Scaling. Validation now compares the "Lip Gap" to the "Face Height," ensuring that mouth-occlusion checks work perfectly whether you are 1 or 5 meters away.

Laptop Perspective Fix: Thresholds for "Straight" poses were widened to 1.4/0.6 to accommodate the physical reality of laptop cameras looking upward at a user's face.

Two-Stage Temporal Filtering: Includes a Moving Average Buffer (5 frames) and a Consensus Gate (3 frames) to eliminate jitter in bad lighting.

3. Acoustic Self-Awareness & Queue Management
Talk-Back Filter: Using difflib.SequenceMatcher, Lumos listens to her own voice. If the microphone hears more than an 80% match to what she is currently speaking, the command is discarded to prevent "Self-Argument" loops.

The "Vision Coma" Fix: Enrollment now triggers an is_paused state. This puts MediaPipe to sleep while the heavy DeepFace brain-build happens, preventing hardware starvation and ensuring the camera feed stays live.

📁 Core Logic Breakdown
🛠️ The Orchestrators
main.py (The CEO): Drives the 30 FPS loop, manages the OpenCV window, and handles manual keyboard overrides ('s', 'f', 'q').

nova_vision_pipeline.py: The heart of the system. Manages the CentroidTracker and the lifecycle of asynchronous recognition threads.

🧠 Biometrics & Identity
nova_pose_validator.py: A stateless geometry engine using MediaPipe FaceLandmarker to verify user poses in 3D space.

nova_enrollment.py: A 5-step guided state machine that verbally directs users through the biometric enrollment process.

nove_face_rec.py: DeepFace-powered recognizer using the Facenet512 model for high-accuracy embedding comparison.

🔊 Audio & Communication
nova_audio.py: A 7-Rank Priority Queue with barge-in capabilities and rank-based cooldowns to prevent greeting spam.

nova_commands.py: A fuzzy-matching NLP engine with a strict Wake Word Gate (Lumo/Lumos) to filter ambient conversations.

🚀 Deployment Instructions
Install Requirements:
pip install opencv-python mediapipe deepface pyttsx3 numpy tf-keras SpeechRecognition

Launch the System:

Terminal 1: python main.py (Starts the Vision/CEO)

Terminal 2: python mock_voice_client.py (Starts the "Ears")

Basic Commands:

"Lumo, Enroll [Name]"

"Lumo, Forget [Name]"

"Lumo, Cancel"

Nova Team | 2026 Graduation Project