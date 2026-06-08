import pickle
import os
import threading
import pyttsx3 as tts

from src.core.nova_logger import logger

# 1. Initialize Voice and Load Brain
def speak(text):
    engine = tts.init()
    engine.say(text)
    engine.runAndWait()

# Configuration
ENCODINGS_FILE = "nova_brain.pkl"

def get_names_from_PK():
    if not os.path.exists(ENCODINGS_FILE):
        logger.warning("The brain file (nova_brain.pkl) does not exist. No names to show.")
        return False
    
    with open(ENCODINGS_FILE, "rb") as f:
        known_faces = pickle.load(f)
    
    return known_faces

def forget_person(name_to_delete, name_dictionary=None):
    # 1. Load the existing brain
    known_faces = {}
    
    if name_dictionary is not None and name_to_delete in name_dictionary.keys():
        known_faces = name_dictionary
        # 4. The Delete command
        del known_faces[name_to_delete]
        
        # 5. Overwrite the old brain file with the updated dictionary
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(known_faces, f)
        announce = f"{name_to_delete}' has been removed from your database."
        return announce
    else:
        announce = f"Could not find '{name_to_delete}' in your database."
        return announce
        

# if __name__ == "__main__":
#     # You can run this file directly from the terminal.
#     # It will ask you who you want to delete.
#     names_stored = get_names_from_PK()
#     if names_stored:
#         print(f"Current people stored in memory: {names_stored}")
#         target_name = input("Enter the EXACT name of the person to forget: ")
#         forget_person(target_name)
