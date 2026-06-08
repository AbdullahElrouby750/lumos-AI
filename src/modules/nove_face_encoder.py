import os
import pickle
from deepface import DeepFace
import threading

from src.core.nova_logger import logger


def build_nova_brain(name = "Person"):
    target_name=name
    # Configuration
    DB_FOLDER = "face_db"
    ENCODINGS_FILE = "nova_brain.pkl"
    success_count = 0
    failure_count = 0
    
    # 1. Initialize or Load the Brain
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            known_faces = pickle.load(f)
    else:
        known_faces = {} # Format: {"Name": [embedding1, embedding2, ...]}

    logger.info("Nova Encoder: Scanning for new images...")

    # 2. Process every image in the folder
    for filename in os.listdir(DB_FOLDER):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # --- BUG X-3 FIX (Part 2): Strict Filtering ---
            # Extract the name from the filename (e.g., 'Adham_0.jpg' -> 'Adham')
            image_owner = filename.split('_')[0]
            
            # If this photo doesn't belong to the person we are enrolling, skip it!
            if image_owner != target_name:
                continue
                
            img_path = os.path.join(DB_FOLDER, filename)
            # ----------------------------------------------

            try:
                logger.info(f"Encoding {name} from {filename}...")
                
                # 'represent' returns a list of dictionaries (one for each face in image)
                # We use 'Facenet512' for high accuracy
                results = DeepFace.represent(img_path=img_path, model_name="Facenet512", enforce_detection=False, max_faces=1, align=True)
                
                if name not in known_faces:
                    known_faces[name] = []
                
                # Store the embedding (the list of numbers)
                known_faces[name].append(results[0]["embedding"])

                # 3. DELETE the image after successful encoding
                os.remove(img_path)
                logger.info(f"Successfully encoded {name} and deleted raw image.")
                success_count += 1

            except Exception as e:
                logger.warning(f"Skipping {filename}: Could not find a clear face. (Keep image for retry)")
                failure_count += 1
                pass
                
    # 4. Final Report
    logger.info(f"Encoding complete. {success_count} faces encoded, {failure_count} failures. {len(known_faces)} unique individuals in the brain.") 

    # 4. Save the mathematical brain to disk
    # --- BUG F-3 FIX: Atomic OS Replace ---
    temp_file = "nova_brain_tmp.pkl"
    final_file = "nova_brain.pkl"
    
    with open(temp_file, "wb") as f:
        pickle.dump(known_faces, f)
        
    # os.replace is an atomic operation at the kernel level.
    # It guarantees the file is never left in a half-written state.
    os.replace(temp_file, final_file)
    
    logger.info("Encoding complete. 'nova_brain.pkl' updated. Memory optimized.")

if __name__ == "__main__":
    build_nova_brain()