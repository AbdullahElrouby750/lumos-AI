import os
import pickle
from deepface import DeepFace
import threading


def build_nova_brain(name = "Person"):
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

    print("Nova Encoder: Scanning for new images...")

    # 2. Process every image in the folder
    for filename in os.listdir(DB_FOLDER):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(DB_FOLDER, filename)
            
            # Extract the name from the filename (e.g., 'Adham_0.jpg' -> 'Adham')
            name = filename.split('_')[0]

            try:
                print(f"Encoding {name} from {filename}...")
                
                # 'represent' returns a list of dictionaries (one for each face in image)
                # We use 'Facenet512' for high accuracy
                results = DeepFace.represent(img_path=img_path, model_name="Facenet512", enforce_detection=False, max_faces=1, align=True)
                
                if name not in known_faces:
                    known_faces[name] = []
                
                # Store the embedding (the list of numbers)
                known_faces[name].append(results[0]["embedding"])

                # 3. DELETE the image after successful encoding
                os.remove(img_path)
                print(f"Successfully encoded {name} and deleted raw image.")
                success_count += 1

            except Exception as e:
                print(f"Skipping {filename}: Could not find a clear face. (Keep image for retry)")
                failure_count += 1
                pass
                
    # 4. Final Report
        print(f"\nEncoding complete. {success_count} faces encoded, {failure_count} failures. {len(known_faces)} unique individuals in the brain.") 

    # 4. Save the mathematical brain to disk
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(known_faces, f)
    
    print(f"\nEncoding complete. 'nova_brain.pkl' updated. Memory optimized.")

if __name__ == "__main__":
    build_nova_brain()