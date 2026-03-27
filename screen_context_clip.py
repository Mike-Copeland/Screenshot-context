import io
import time
import ctypes
import threading
from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Optional

import cv2
import mss
import numpy as np
import keyboard
from PIL import Image, ImageDraw, ImageFont

import win32clipboard


# =========================
# CONFIG
# =========================

HOTKEY_FAST = "ctrl+r"
HOTKEY_RICH = "ctrl+shift+r"

BUFFER_SECONDS = 10
CAPTURE_FPS = 1
CAPTURE_MONITOR_INDEX = 1

JPEG_QUALITY = 70

# FAST = lighter/fewer frames
FAST_MAX_FRAMES = 2

# RICH = more context
RICH_MAX_FRAMES = 4

# Output image sizing
FRAME_PREVIEW_WIDTH = 700
PANEL_BG = (18, 18, 18)
TEXT_COLOR = (235, 235, 235)
SUBTEXT_COLOR = (180, 180, 180)
BORDER_COLOR = (60, 60, 60)
LABEL_BG = (35, 35, 35)

TOP_MARGIN = 16
SIDE_MARGIN = 16
GAP = 12
LABEL_HEIGHT = 28
BOTTOM_MARGIN = 16

PASTE_DELAY_SECONDS = 0.08


# =========================
# WINDOWS HELPERS
# =========================

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


# =========================
# CLIPBOARD IMAGE PASTE
# =========================

def copy_image_to_clipboard(img: Image.Image) -> None:
    """
    Copies a PIL image to the Windows clipboard as CF_DIB.
    Most desktop apps that support image paste will accept this.
    """
    output = io.BytesIO()

    # CF_DIB expects BMP bytes without the 14-byte BMP file header.
    bmp = img.convert("RGB")
    bmp.save(output, "BMP")
    data = output.getvalue()[14:]

    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
    finally:
        win32clipboard.CloseClipboard()


def paste_clipboard_to_active_target() -> None:
    time.sleep(PASTE_DELAY_SECONDS)
    keyboard.press_and_release("ctrl+v")


# =========================
# FRAME MODEL
# =========================

@dataclass
class FrameRecord:
    ts: float
    jpeg_bytes: bytes
    thumb_gray: np.ndarray


# =========================
# ROLLING BUFFER
# =========================

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
                raise RuntimeError(
                    f"Monitor index {self.monitor_index} not available. "
                    f"Detected monitors: {len(monitors) - 1}"
                )

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


# =========================
# FRAME SCORING / SELECTION
# =========================

def diff_scores(records: List[FrameRecord]) -> List[float]:
    if len(records) < 2:
        return [0.0] * len(records)

    scores = [0.0]
    for i in range(1, len(records)):
        d = cv2.absdiff(records[i - 1].thumb_gray, records[i].thumb_gray)
        scores.append(float(np.mean(d)))
    return scores


def select_fast_frames(records: List[FrameRecord], max_frames: int = FAST_MAX_FRAMES) -> List[FrameRecord]:
    """
    FAST tier:
    - choose the strongest recent change
    - choose final frame
    """
    if len(records) <= max_frames:
        return records

    scores = diff_scores(records)
    n = len(records)

    selected = {n - 1}

    late_start = max(1, n // 2)
    late_idx = max(range(late_start, n), key=lambda i: scores[i])
    selected.add(late_idx)

    return [records[i] for i in sorted(selected)[:max_frames]]


def select_rich_frames(records: List[FrameRecord], max_frames: int = RICH_MAX_FRAMES) -> List[FrameRecord]:
    """
    RICH tier:
    - early anchor
    - strongest change in first half
    - strongest change in second half
    - final frame
    """
    if len(records) <= max_frames:
        return records

    scores = diff_scores(records)
    n = len(records)
    half = max(1, n // 2)

    selected = {0, n - 1}

    if half > 1:
        first_half_idx = max(range(1, half), key=lambda i: scores[i], default=0)
        selected.add(first_half_idx)

    if n - half > 1:
        second_half_idx = max(range(half, n), key=lambda i: scores[i], default=n - 1)
        selected.add(second_half_idx)

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


# =========================
# IMAGE COMPOSITION
# =========================

def load_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def decode_frame_to_pil(record: FrameRecord) -> Image.Image:
    arr = np.frombuffer(record.jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def fit_width(img: Image.Image, width: int) -> Image.Image:
    w, h = img.size
    if w == width:
        return img
    scale = width / w
    new_h = int(h * scale)
    return img.resize((width, new_h), Image.LANCZOS)


def compose_panel(records: List[FrameRecord], tier_name: str, active_window_title: str) -> Image.Image:
    title_font = load_font(20)
    meta_font = load_font(14)
    label_font = load_font(14)

    decoded = [fit_width(decode_frame_to_pil(r), FRAME_PREVIEW_WIDTH) for r in records]

    total_height = TOP_MARGIN + 36 + 24 + GAP
    for img in decoded:
        total_height += LABEL_HEIGHT + img.height + GAP
    total_height += BOTTOM_MARGIN - GAP

    panel_width = SIDE_MARGIN * 2 + FRAME_PREVIEW_WIDTH
    panel = Image.new("RGB", (panel_width, total_height), PANEL_BG)
    draw = ImageDraw.Draw(panel)

    y = TOP_MARGIN

    header = f"Screen Context Paste — {tier_name}"
    draw.text((SIDE_MARGIN, y), header, font=title_font, fill=TEXT_COLOR)
    y += 30

    meta = f"Window: {active_window_title or '(unknown)'}"
    draw.text((SIDE_MARGIN, y), meta, font=meta_font, fill=SUBTEXT_COLOR)
    y += 22

    for idx, (rec, img) in enumerate(zip(records, decoded), start=1):
        ts_str = time.strftime("%H:%M:%S", time.localtime(rec.ts))

        draw.rounded_rectangle(
            (SIDE_MARGIN, y, panel_width - SIDE_MARGIN, y + LABEL_HEIGHT),
            radius=6,
            fill=LABEL_BG,
            outline=BORDER_COLOR,
            width=1,
        )
        label = f"Frame {idx} — {ts_str}"
        draw.text((SIDE_MARGIN + 10, y + 6), label, font=label_font, fill=TEXT_COLOR)
        y += LABEL_HEIGHT

        panel.paste(img, (SIDE_MARGIN, y))
        draw.rectangle(
            (SIDE_MARGIN, y, SIDE_MARGIN + img.width, y + img.height),
            outline=BORDER_COLOR,
            width=1,
        )
        y += img.height + GAP

    return panel


# =========================
# APP
# =========================

class App:
    def __init__(self):
        self.buffer = RollingScreenBuffer(
            buffer_seconds=BUFFER_SECONDS,
            fps=CAPTURE_FPS,
            monitor_index=CAPTURE_MONITOR_INDEX,
            jpeg_quality=JPEG_QUALITY,
        )
        self._busy = threading.Lock()

    def start(self) -> None:
        self.buffer.start()
        keyboard.add_hotkey(HOTKEY_FAST, lambda: self.trigger("FAST"))
        keyboard.add_hotkey(HOTKEY_RICH, lambda: self.trigger("RICH"))

    def stop(self) -> None:
        self.buffer.stop()

    def trigger(self, tier: str) -> None:
        if not self._busy.acquire(blocking=False):
            beep_error()
            return

        threading.Thread(target=self._run, args=(tier,), daemon=True).start()

    def _run(self, tier: str) -> None:
        try:
            records = self.buffer.snapshot()
            if len(records) < 2:
                raise RuntimeError("Not enough buffered frames yet.")

            active_window = get_active_window_title()

            if tier == "FAST":
                selected = select_fast_frames(records)
            else:
                selected = select_rich_frames(records)

            panel = compose_panel(selected, tier_name=tier, active_window_title=active_window)
            copy_image_to_clipboard(panel)
            paste_clipboard_to_active_target()

            print(f"[{tier}] pasted composed image with {len(selected)} frame(s)")
            beep_ok()

        except Exception as e:
            print(f"[ERROR] {e}")
            beep_error()

        finally:
            self._busy.release()


# =========================
# MAIN
# =========================

def main() -> None:
    print("Starting Screen Context Paste...")
    print(f"FAST hotkey: {HOTKEY_FAST}")
    print(f"RICH hotkey: {HOTKEY_RICH}")
    print(f"Rolling buffer: {BUFFER_SECONDS}s @ {CAPTURE_FPS} fps")
    print("This tool pastes a composed screenshot panel into the active target.")
    print("Press ESC to exit.")

    app = App()
    app.start()

    try:
        keyboard.wait("esc")
    finally:
        app.stop()


if __name__ == "__main__":
    main()
