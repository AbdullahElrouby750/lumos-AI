import json
import socket
import threading

class CommandListener(threading.Thread):
    """Listens for local JSON voice commands on UDP and pushes them into a queue."""

    def __init__(self, command_queue, voice_queue, host="127.0.0.1", port=65432):
        super().__init__(daemon=True)
        self.command_queue = command_queue
        self.voice_queue = voice_queue
        self.host = host
        self.port = port
        self.running = True
        self.sock = None
        self.noise_floor = 0.5 # Default

    def run(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.host, self.port))
            self.sock.settimeout(1.0)
            print(f"Command listener bound to {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to start command listener: {e}")
            self.running = False
            return

        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
                payload = json.loads(data.decode("utf-8"))
                raw_text = payload.get("raw_text", "")
                # Import here to avoid circular import
                from nova_commands import parse_intent
                command = parse_intent(raw_text)
                if command["intent"] != "INTENT_NONE":
                    command["raw_text"] = raw_text
                    self.command_queue.put(command)
                    print(f"Received voice command: {command}")
    
                    # Voice barge-in: Purge old messages and PAUSE the worker
                    if hasattr(self.voice_queue, 'clear_below_critical'):
                        self.voice_queue.clear_below_critical()
                        self.voice_queue.pause_below_critical()
            except socket.timeout:
                continue
            except json.JSONDecodeError:
                print("Received invalid JSON command payload.")
            except Exception as e:
                print(f"Command listener error: {e}")

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass