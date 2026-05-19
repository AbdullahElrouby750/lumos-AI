# Phase 2 (Part 2): Implementation Instructions

## 1. Handling the `_temp` Files (CRITICAL)
- Remember that any file ending in `_temp.py` is for reference only. 
- You must import the unified models from `nova_network_models.py` and the server from `nova_server.py`.
- Use the `VoiceQueue` constants from `nova_audio.py` for event priorities.

## 2. The Command Queue
- The `NovaAIWorker` class must instantiate its own `queue.Queue`.
- The `_worker_loop` should use a blocking `get()` with a timeout (e.g., `get(timeout=1.0)`) so the thread sleeps efficiently while waiting for commands.

## 3. Graceful Network Failures
- If `brain_module.py` raises an exception (e.g., no internet connection to reach Google Gemini), the worker must catch it.
- Upon failure, it must send a `SpeakEvent` (or `BaseEvent`) back to the user saying: *"Network connection lost. Cannot process the request right now."* with `PRIORITY_WARNING`.

## 4. Style Guidelines
- Maintain strict Python type hinting.
- Use `logging.getLogger(__name__)` for all status outputs. No print statements.