# Screen Context Paste

A Windows-first desktop helper that keeps a local rolling screenshot buffer for the last 10 seconds of screen activity. When you press a hotkey, it selects representative screenshots from that recent buffer and pastes them directly into the active window, field, cell, or prompt target, similar to `Ctrl+V` but with recent visual context instead of plain clipboard text.

## What it does

- `Ctrl+R` = **FAST mode**
  - Selects a compact set of representative screenshots from the recent rolling buffer.
  - Pastes them directly into the active destination.
  - Intended for the quickest, lowest-friction visual context paste.

- `Ctrl+Shift+R` = **RICH mode**
  - Selects a richer set of representative screenshots from the recent rolling buffer.
  - Pastes them directly into the active destination.
  - Intended for cases where more recent navigation/context is needed.

## Why this exists

A single screenshot often is not enough. In real workflows, you may click around for several seconds to find the right window, tab, pane, table, or field before pasting into a target. This tool preserves that short navigation window and pastes the recent visual context directly where you need it.

## Core idea

This tool acts like an augmented visual `Ctrl+V`.

Instead of pasting only the current clipboard contents, it pastes selected screenshots from the last several seconds of recent screen history into the active target.

## Requirements

- Windows
- Python 3.9+

Install dependencies:

```bash
pip install -U mss opencv-python numpy keyboard pyperclip pillow
