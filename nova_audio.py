import queue
import threading
import time
import pyttsx3

class VoiceQueue:
    """
    Thread-safe voice queue for managing TTS output.
    Uses a background worker thread to consume voice requests sequentially,
    preventing pyttsx3 driver crashes from concurrent calls.
    
    Features:
    - Priority ranking: DANGER > WARNING > SOCIAL > INFO
    - Time-to-Live (TTL): Messages older than 2.0 seconds are discarded
    - Non-blocking: Main thread never waits for TTS operation
    """

    # Priority constants (lower number = higher priority)
    PRIORITY_DANGER = 1
    PRIORITY_WARNING = 2
    PRIORITY_SOCIAL = 3
    PRIORITY_INFO = 4

    def __init__(self):
        """Initialize the VoiceQueue and start the background worker thread."""
        self.queue = queue.PriorityQueue()
        self.running = True
        self.lock = threading.Lock()

        # Start the background worker thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def speak(self, text, priority=None):
        """
        Add text to the voice queue with a priority level.
        
        Args:
            text (str): The text to speak
            priority (int): Priority level (DANGER=1, WARNING=2, SOCIAL=3, INFO=4)
                           Default is PRIORITY_INFO
        """
        if priority is None:
            priority = self.PRIORITY_INFO

        timestamp = time.time()
        try:
            self.queue.put((priority, timestamp, text), block=False)
        except queue.Full:
            print(f"Voice queue is full. Discarding: {text}")

    def _worker(self):
        """
        Background worker thread. 
        Creates a fresh TTS engine per sentence to bypass Windows SAPI5 thread locks.
        """
        while self.running:
            try:
                # Wait for a message (0.5s timeout keeps the loop checking for self.running)
                priority, timestamp, text = self.queue.get(timeout=0.5)

                # Check TTL: Increased to 15 seconds to handle long spam tests safely
                if time.time() - timestamp > 15.0:
                    print(f"[VoiceQueue] Discarded outdated message: {text}")
                    continue

                # --- THE FIX ---
                # Initialize, speak, and destroy the engine in one clean sweep
                engine = pyttsx3.init()
                engine.setProperty('rate', 190)
                engine.say(text)
                engine.runAndWait()
                del engine # Force Windows to release the audio driver lock

            except queue.Empty:
                pass # Queue is empty, just loop again
            except Exception as e:
                print(f"[VoiceQueue] Worker error: {e}")

    def stop(self):
        """Stop the background worker gracefully."""
        with self.lock:
            self.running = False
        self.worker_thread.join(timeout=5.0)
        # We removed the self.engine.stop() here because the engine deletes itself now!

    def get_queue_size(self):
        """Return the current size of the voice queue."""
        return self.queue.qsize()


# Global instance (optional, for convenience)
_voice_queue_instance = None

def get_voice_queue():
    """Get or create the global VoiceQueue instance."""
    global _voice_queue_instance
    if _voice_queue_instance is None:
        _voice_queue_instance = VoiceQueue()
    return _voice_queue_instance


# ===== TEST / DEMO CODE =====
if __name__ == "__main__":
    print("Testing VoiceQueue...")
    
    voice_queue = VoiceQueue()
    
    # Test 1: Simple speak
    print("\n[Test 1] Speaking 'Hello World'...")
    voice_queue.speak("Hello World", VoiceQueue.PRIORITY_INFO)
    time.sleep(2)
    
    # Test 2: Priority ranking
    print("\n[Test 2] Testing priority ranking (add 3 messages with different priorities)...")
    voice_queue.speak("This is info", VoiceQueue.PRIORITY_INFO)
    voice_queue.speak("This is a warning", VoiceQueue.PRIORITY_WARNING)
    voice_queue.speak("This is danger", VoiceQueue.PRIORITY_DANGER)
    time.sleep(5)
    
    # Test 3: Spam test (rapid fire)
    print("\n[Test 3] Spam test - adding 10 messages rapidly...")
    for i in range(10):
        voice_queue.speak(f"Message number {i + 1}", VoiceQueue.PRIORITY_INFO)
    
    print(f"Queue size: {voice_queue.get_queue_size()}")
    time.sleep(15)  # Wait for all messages to be spoken
    
    # Test 4: TTL test (old messages)
    print("\n[Test 4] TTL test - manually adding old message...")
    old_timestamp = time.time() - 3.0  # 3 seconds old
    voice_queue.queue.put((VoiceQueue.PRIORITY_INFO, old_timestamp, "This should be discarded"))
    time.sleep(2)
    
    print("\n[Test Complete] Cleaning up...")
    voice_queue.stop()
    print("VoiceQueue stopped successfully!")
