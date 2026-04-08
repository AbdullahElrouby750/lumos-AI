import argparse
import json
import socket
import time

import speech_recognition as sr

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 65432


def send_payload(sock, host, port, phrase):
    payload = {
        "wake_word": True,
        "raw_text": phrase,
        "source": "laptop"
    }
    data = json.dumps(payload).encode("utf-8")
    sock.sendto(data, (host, port))
    print(f"Sent payload: {payload}")


def listen_and_send(host, port):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        print(f"Mock voice client listening on mic and sending commands to {host}:{port}")
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            while True:
                print("Listening for voice command...")
                try:
                    audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
                    phrase = recognizer.recognize_google(audio)
                    phrase = phrase.strip()

                    if not phrase:
                        continue

                    print(f"Recognized phrase: {phrase}")
                    send_payload(sock, host, port, phrase)
                    time.sleep(0.5)

                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    print("Could not understand audio.")
                except sr.RequestError as e:
                    print(f"Speech recognition request failed: {e}")
                    time.sleep(1)
                except KeyboardInterrupt:
                    print("Exiting mock voice client.")
                    break
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Mock voice client sending commands to Lumos.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Local host to send commands to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Local UDP port to send commands to")
    args = parser.parse_args()

    listen_and_send(args.host, args.port)


if __name__ == "__main__":
    main()
