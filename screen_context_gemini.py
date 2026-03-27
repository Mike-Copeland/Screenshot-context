import os, io, time, ctypes, threading
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timezone
from typing import Deque, List, Optional

import cv2
import keyboard
import mss
import numpy as np
import pyperclip
from PIL import Image
from google import genai

HOTKEY_PASTE = "ctrl+r"
HOTKEY_CONTEXT = "ctrl+shift+r"
BUFFER_SECONDS = 10
CAPTURE_FPS = 1
MAX_SELECTED_FRAMES = 4
JPEG_QUALITY = 70
CAPTURE_MONITOR_INDEX = 1
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
USE_CLIPBOARD_AS_LIVE_PROMPT = True
MAX_CLIPBOARD_PROMPT_CHARS = 4000
PASTE_RESULT_IN_PASTE_MODE = True
PASTE_CONTEXT_IN_CONTEXT_MODE = False

PROMPT_PASTE = (
    "Use these recent screen frames as context for the task I was just performing. "
    "Infer what text should be inserted into the active field or spreadsheet cell. "
    "Infer intent from the sequence of screens, not only the final state. "
    "Use only visible evidence from the frames and the user note. "
    "If ambiguous, return exactly: UNCERTAIN. Return only the final text."
)

PROMPT_CONTEXT = (
    "Use these recent screen frames as context for the task I was just performing. "
    "Return a compact context block that I can prepend to my next prompt. "
    "Focus on what I was navigating toward, what became visible, and the likely target task. "
    "Use only visible evidence from the frames and the user note. "
    "If ambiguous, return exactly: UNCERTAIN. Be concise."
)


def get_active_window_title() -> str:
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ""
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def beep_ok() -> None:
    try:
        ctypes.windll.user32.MessageBeep(0x00000040)
    except Exception:
        pass


def beep_error() -> None:
    try:
        ctypes.windll.user32.MessageBeep(0x00000010)
    except Exception:
        pass


@dataclass
class FrameRecord:
    ts: float
    jpeg_bytes: bytes
    thumb_gray: np.ndarray


class RollingScreenBuffer:
    def __init__(self, buffer_seconds: int, fps: int, monitor_index: int, jpeg_quality: int):
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.monitor_index = monitor_index
        self.jpeg_quality = jpeg_quality
        self.frames: Deque[FrameRecord] = deque(maxlen=max(2, buffer_seconds * fps))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def snapshot(self) -> List[FrameRecord]:
        with self._lock:
            return list(self.frames)

    def _loop(self) -> None:
        interval = 1.0 / self.fps
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        with mss.mss() as sct:
            monitors = sct.monitors
            if self.monitor_index >= len(monitors):
                raise RuntimeError(f"Monitor index {self.monitor_index} not available.")
            monitor = monitors[self.monitor_index]
            while not self._stop.is_set():
                t0 = time.time()
                raw = np.array(sct.grab(monitor))
                bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                ok, enc = cv2.imencode(".jpg", bgr, encode_params)
                if ok:
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    thumb = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
                    rec = FrameRecord(ts=t0, jpeg_bytes=enc.tobytes(), thumb_gray=thumb)
                    with self._lock:
                        self.frames.append(rec)
                elapsed = time.time() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)


def read_clipboard_text() -> str:
    if not USE_CLIPBOARD_AS_LIVE_PROMPT:
        return ""
    try:
        text = pyperclip.paste()
        if not isinstance(text, str):
            return ""
        text = text.strip()
        if not text:
            return ""
        return text[:MAX_CLIPBOARD_PROMPT_CHARS]
    except Exception:
        return ""


def diff_scores(records: List[FrameRecord]) -> List[float]:
    if len(records) < 2:
        return [0.0] * len(records)
    scores = [0.0]
    for i in range(1, len(records)):
        d = cv2.absdiff(records[i - 1].thumb_gray, records[i].thumb_gray)
        scores.append(float(np.mean(d)))
    return scores


def select_representative_frames(records: List[FrameRecord], max_frames: int = MAX_SELECTED_FRAMES) -> List[FrameRecord]:
    if len(records) <= max_frames:
        return records

    scores = diff_scores(records)
    n = len(records)
    selected = {0}
    half = n // 2

    if half > 1:
        first_half_idx = max(range(1, max(2, half)), key=lambda i: scores[i])
        selected.add(first_half_idx)

    if n - half > 1:
        second_half_idx = max(range(max(half, 1), n), key=lambda i: scores[i])
        selected.add(second_half_idx)

    if n - 1 not in selected:
        last = n - 1
        if scores[last] > 1.0 or len(selected) < max_frames:
            selected.add(last)

    if len(selected) < max_frames:
        candidates = sorted(range(1, n - 1), key=lambda i: scores[i], reverse=True)
        for idx in candidates:
            if idx in selected:
                continue
            if any(abs(idx - s) <= 1 for s in selected):
                continue
            selected.add(idx)
            if len(selected) >= max_frames:
                break

    return [records[i] for i in sorted(selected)[:max_frames]]


class GeminiClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = genai.Client()

    def run(self, mode: str, frames: List[FrameRecord], metadata: dict, live_prompt: str) -> str:
        base_prompt = PROMPT_PASTE if mode == "PASTE" else PROMPT_CONTEXT
        prompt = (
            f"MODE: {mode}\n"
            f"UTC timestamp: {metadata['timestamp_utc']}\n"
            f"Active window title: {metadata['active_window_title']}\n"
            f"Captured seconds: {metadata['captured_seconds']}\n"
            f"Frame count: {metadata['frame_count']}\n\n"
            f"Base instruction:\n{base_prompt}\n\n"
            f"User note / live prompt:\n{live_prompt or '(none)'}"
        )

        contents = [prompt]
        for fr in frames:
            contents.append(Image.open(io.BytesIO(fr.jpeg_bytes)))

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
        )

        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Empty Gemini response.")
        return text


def paste_text(text: str) -> None:
    pyperclip.copy(text)
    time.sleep(0.05)
    keyboard.press_and_release("ctrl+v")


class App:
    def __init__(self):
        self.buffer = RollingScreenBuffer(BUFFER_SECONDS, CAPTURE_FPS, CAPTURE_MONITOR_INDEX, JPEG_QUALITY)
        self.gemini = GeminiClient(MODEL_NAME)
        self._busy = threading.Lock()

    def start(self) -> None:
        self.buffer.start()
        keyboard.add_hotkey(HOTKEY_PASTE, lambda: self.trigger("PASTE"))
        keyboard.add_hotkey(HOTKEY_CONTEXT, lambda: self.trigger("CONTEXT"))

    def stop(self) -> None:
        self.buffer.stop()

    def trigger(self, mode: str) -> None:
        if not self._busy.acquire(blocking=False):
            beep_error()
            return
        threading.Thread(target=self._run, args=(mode,), daemon=True).start()

    def _run(self, mode: str) -> None:
        try:
            records = self.buffer.snapshot()
            if len(records) < 2:
                raise RuntimeError("Not enough buffered frames yet.")

            selected = select_representative_frames(records)
            metadata = {
                "active_window_title": get_active_window_title(),
                "captured_seconds": BUFFER_SECONDS,
                "frame_count": len(selected),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

            live_prompt = read_clipboard_text()
            result = self.gemini.run(mode, selected, metadata, live_prompt)

            if mode == "PASTE" and PASTE_RESULT_IN_PASTE_MODE:
                paste_text(result)
            else:
                pyperclip.copy(result)
                if mode == "CONTEXT" and PASTE_CONTEXT_IN_CONTEXT_MODE:
                    time.sleep(0.05)
                    keyboard.press_and_release("ctrl+v")

            print(f"[{mode}] {result}")
            beep_ok()

        except Exception as e:
            print(f"[ERROR] {e}")
            beep_error()
        finally:
            self._busy.release()


def main() -> None:
    print("Starting screen context Gemini helper...")
    print(f"Hotkey paste:   {HOTKEY_PASTE}")
    print(f"Hotkey context: {HOTKEY_CONTEXT}")
    print(f"Buffer: {BUFFER_SECONDS}s @ {CAPTURE_FPS} fps")
    print(f"Model:  {MODEL_NAME}")
    print("Live prompt source: clipboard text")
    print("Press ESC to exit.")

    app = App()
    app.start()
    try:
        keyboard.wait("esc")
    finally:
        app.stop()


if __name__ == "__main__":
    main()
