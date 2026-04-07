# nova_audio.py  — full replacement

import hashlib
import queue
import threading
import time
import pyttsx3


class VoiceQueue:
    """
    Thread-safe TTS queue with a 7-rank priority hierarchy, barge-in,
    pause/resume, and per-rank cooldown spam control.

    Priority ranks (lower = higher priority):
      1  PRIORITY_CRITICAL   – immediate physical danger
      2  PRIORITY_FEEDBACK   – command acknowledgements ("Cancelled", "Are you sure?")
      3  PRIORITY_SYSTEM     – enrollment / forget instructions
      4  PRIORITY_WARNING    – velocity-based approaching alerts
      5  PRIORITY_SOCIAL     – face identification ("I see Adham")
      6  PRIORITY_INFO       – status updates (battery, startup)
      7  PRIORITY_LOW        – background descriptions
    """

    PRIORITY_CRITICAL = 1
    PRIORITY_FEEDBACK = 2
    PRIORITY_SYSTEM   = 3
    PRIORITY_WARNING  = 4
    PRIORITY_SOCIAL   = 5
    PRIORITY_INFO     = 6
    PRIORITY_LOW      = 7

    # Per-rank spam cooldown in seconds (0 = no cooldown)
    RANK_COOLDOWNS = {
        5: 10.0,   # SOCIAL: same face/object not re-announced for 10 s
    }

    TTL = 7.0  # seconds

    def __init__(self):
        self._queue      = queue.PriorityQueue()
        self._running    = True
        self._lock       = threading.Lock()
        self._is_paused  = threading.Event()   # set → worker holds non-CRITICAL items
        self._is_speaking = threading.Event()  # set → TTS engine is active right now
        self._cooldowns  = {}                  # {(priority, text_hash): last_spoken_time}

        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str, priority: int = None) -> bool:
        """
        Enqueue text for TTS. Returns False if suppressed by cooldown.
        Priority defaults to PRIORITY_LOW.
        """
        if priority is None:
            priority = self.PRIORITY_LOW

        # Cooldown check (only for ranks that define one)
        cooldown = self.RANK_COOLDOWNS.get(priority, 0)
        if cooldown > 0:
            key = (priority, hashlib.md5(text.encode()).hexdigest())
            last = self._cooldowns.get(key, 0)
            if time.time() - last < cooldown:
                return False
            self._cooldowns[key] = time.time()

        self._queue.put((priority, time.time(), text))
        return True

    def clear_below_critical(self):
        """
        Purge all queued items except PRIORITY_CRITICAL (rank 1).
        Call on voice barge-in.
        """
        self._drain_ranks(range(2, 8))

    def pause_below_critical(self):
        """
        Tell the worker to hold any non-CRITICAL item in place.
        Call when a voice command is being processed.
        """
        self._is_paused.set()

    def resume(self):
        """Resume normal queue processing after a command completes."""
        self._is_paused.clear()

    def cancel_queues(self, *ranks):
        """
        Drain specific ranks from the queue.
        e.g. cancel_queues(PRIORITY_SYSTEM, PRIORITY_INFO) for Cancel command.
        """
        self._drain_ranks(ranks)

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking.is_set()

    def get_queue_size(self) -> int:
        return self._queue.qsize()

    def stop(self):
        with self._lock:
            self._running = False
        self._worker_thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_ranks(self, ranks_to_remove):
        """Rebuild the queue, discarding items whose priority is in ranks_to_remove."""
        keep = []
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item[0] not in ranks_to_remove:
                    keep.append(item)
            except queue.Empty:
                break
        fresh = queue.PriorityQueue()
        for item in keep:
            fresh.put(item)
        self._queue = fresh

    def _worker(self):
        while self._running:
            try:
                priority, timestamp, text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # TTL check
            if time.time() - timestamp > self.TTL:
                print(f"[VoiceQueue] TTL expired, discarding: {text!r}")
                continue

            # Pause check: non-CRITICAL items wait while paused
            if self._is_paused.is_set() and priority > self.PRIORITY_CRITICAL:
                # Re-queue with original timestamp so it doesn't expire unfairly
                self._queue.put((priority, timestamp, text))
                time.sleep(0.1)
                continue

            # Speak
            try:
                self._is_speaking.set()
                engine = pyttsx3.init()
                engine.setProperty('rate', 200)
                engine.say(text)
                engine.runAndWait()
                del engine
            except Exception as e:
                print(f"[VoiceQueue] TTS error: {e}")
            finally:
                self._is_speaking.clear()


# Global singleton accessor
_instance = None

def get_voice_queue() -> VoiceQueue:
    global _instance
    if _instance is None:
        _instance = VoiceQueue()
    return _instance