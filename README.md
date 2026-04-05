
# Lumos: Social Vision & Face Recognition Module (Nova Team)

## 👁️ Project Overview
This repository contains the **Social Vision Module** for **Lumos**, an assistive technology wearable designed for the visually impaired. 

Moving beyond standard obstacle detection, this module acts as a "Social Brain." It actively scans the environment, detects human faces, calculates their distance, identifies known individuals using Deep Learning, and provides real-time, context-aware audio feedback to the user.

### The Prime Directive: Zero Blocking
Because this is assistive technology for the visually impaired, **camera freezing or frame dropping is a critical safety hazard**. This entire architecture is built around a "Prime Directive": The main camera loop must *never* drop below 30 FPS. All heavy processing (Facenet embeddings, Text-to-Speech audio drivers, network listening, and state machines) are offloaded to asynchronous background threads and queues.

---

## ⚙️ How It Works (The Architecture)
The system uses a **Microservice-style CEO/Worker Architecture**:
1. **The CEO (`main.py`):** Captures frames at 30 FPS and delegates them.
2. **The Vision Pipeline:** Uses a lightweight `CentroidTracker` to follow moving bounding boxes. When a *new* ID enters the frame, it sends a cropped image to a background thread to run the heavy Facenet512 math.
3. **The Spam Filter & Collision Zone:** To prevent the AI from constantly repeating names, it uses "Name-Level Cooldowns" (30 seconds). It also calculates a central 40% "Collision Zone," ignoring people passing safely on the left or right, and only alerting the user if a recognized face is expanding in size (approaching).
4. **The NLP Engine:** Uses a UDP socket to listen for voice commands (e.g., "Hey Lumo, enroll Adham").
5. **The Audio Manager:** A thread-safe Priority Queue ensures the Text-to-Speech engine never crashes, dropping outdated messages and allowing critical alerts (DANGER) to interrupt standard info.

---

## 📁 Module Breakdown

### 1. Core Orchestration
#### `main.py` (The CEO)
The entry point of the application. It initializes the camera, spins up the background threads (Audio Queue, Command Listener), and runs the master `while` loop. It routes frames to the Vision Pipeline and the Enrollment Manager. It also contains lightweight keyboard overrides (`s` to enroll, `f` to forget, `q` to quit) that spawn temporary daemon threads to prevent `input()` from blocking the camera.

#### `utils.py` (Math & Tracking)
Contains pure mathematical logic and drawing utilities to keep the main loops clean.
* `CentroidTracker`: A custom class that calculates Euclidean distance between bounding box centers across frames to assign and maintain temporary IDs for faces. This prevents the "amnesia" effect when a face turns slightly.
* `calculate_distance(face_width)`: Uses a calibrated focal length (678.57) and Triangle Similarity to estimate how many meters away a person is.
* `is_in_collision_zone()` / `is_bbox_expanding()`: Calculates if a face is in the center 40% of the user's view and if they are taking a step toward the user.

### 2. Vision & Recognition
#### `nova_vision_pipeline.py` (The Security Guard)
Orchestrates the detection and recognition process.
* **How it works:** Receives a frame, calls the detector, updates the Centroid Tracker, and handles the Async Recognition. It manages the `name_cooldowns` memory dictionary and triggers `voice_queue.speak()` based on the Collision Zone logic.

#### `nova_face_detector.py`
A lightweight wrapper around Google's `MediaPipe` Face Detector (`blaze_face_short_range.tflite`). Built as a class to maintain the detector's state without constantly reloading it. Returns bounding boxes.

#### `nove_face_rec.py`
The recognition engine utilizing `DeepFace` and the `Facenet512` model.
* **How it works:** Loads `nova_brain.pkl` into memory. When called via `recognize_face_crop()`, it resizes the input to 160x160, extracts the mathematical embedding, and calculates the Cosine Distance against the known database. Returns a name or "Unknown" based on a strict 0.6 distance threshold.

### 3. Identity Management (The Brain)
#### `nova_enrollment.py` (The Guided State Machine)
Handles the addition of new faces to the system completely asynchronously.
* **How it works:** When triggered, it acts as a state machine that verbally guides the user through a biometric scan ("Look straight", "Look left", "Look right", etc.). It waits 3 seconds between prompts, saves 5 diverse angles to a `face_db` folder, and then triggers the Encoder.

#### `nove_face_encoder.py`
* `build_nova_brain(name)`: Scans the `face_db` folder, uses DeepFace to extract 512-dimensional embeddings for the newly captured images, appends them to `nova_brain.pkl`, and deletes the raw images to protect user privacy.

#### `nove_forget.py`
* `forget_person(name)`: Safely loads the `.pkl` brain, deletes the target's embeddings from the dictionary, and overwrites the file, effectively giving the AI localized amnesia for that individual.

### 4. Audio & Voice Input
#### `nova_audio.py` (The Voice Queue)
A bulletproof, thread-safe Producer-Consumer queue for `pyttsx3`.
* **How it works:** Prevents the notorious Windows SAPI5 thread-lock bug by initializing and destroying the TTS engine *inside* the worker loop for every sentence. 
* **Features:** Supports Priorities (1-4). Includes a Time-to-Live (TTL) of 15 seconds to drop stale messages, and features a `clear_non_danger_queue()` method for instant voice barge-in.

#### `nova_listener.py` (The Ear)
A UDP socket server running on a background daemon thread (Port `65432`). 
* **How it works:** Listens for JSON payloads containing transcribed speech from the mobile app (or mock client). Pushes valid commands into a queue for `main.py` to execute.

#### `nova_commands.py` (The NLP Engine)
* `parse_intent(raw_text)`: A custom Natural Language Processor. It strips filler words, isolates keywords (enroll, forget, quit), extracts the target entity (the person's name), and returns a structured intent dictionary (e.g., `INTENT_FORGET`, Target: `Adham`).

#### `mock_voice_client.py` (The Mouth)
A standalone testing script that simulates the future Flutter mobile application.
* **How it works:** Uses the `speech_recognition` library to listen to the laptop microphone, converts speech to text, packages it into the Lumos JSON format, and fires it over UDP to `nova_listener.py`.

---

## 🚀 How to Run

1. **Install Dependencies:**
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install opencv-python mediapipe deepface pyttsx3 numpy SpeechRecognition
   ```
2. **Start the Main Vision Loop:**
   ```bash
   python main.py
   ```
3. **Start the Voice Client (In a separate terminal):**
   ```bash
   python mock_voice_client.py
   ```
4. **Usage:**
   * Speak into the mock client: *"Hey Lumo, enroll [Name]"* to start the guided setup.
   * Speak: *"Hey Lumo, forget [Name]"* to delete them.
   * Speak: *"Quit"* to shut down the system gracefully.
   * Alternatively, use keyboard overrides in the video window (`s` to save, `f` to forget, `q` to quit).
```
