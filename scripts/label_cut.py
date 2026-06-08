"""Interactive relabeling tool for cut_to_label == 2 rows."""
from __future__ import annotations

import csv
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "dataset" / "final_labeled_260526.csv"
IMG_ROOT = PROJECT_ROOT / "dataset" / "classification" / "crops_25pct"
CSV_ENCODING = "cp949"

# ── label map ────────────────────────────────────────────────────────────────
LABEL_MAP: dict[str, int] = {
    "cut": 0,
    "danger": 1,
    "excluded": 2,
    "UK": 3,
}
UNLABELED_VALUE = "2"  # cut_to_label == 2  →  needs review

# ── display ──────────────────────────────────────────────────────────────────
MAX_IMG_SIZE = 800


def _find_image(original_label: str, fname: str) -> Path | None:
    """Return image path; search original_label folder first, then all subfolders."""
    candidate = IMG_ROOT / original_label / fname
    if candidate.exists():
        return candidate
    for subdir in IMG_ROOT.iterdir():
        p = subdir / fname
        if p.exists():
            return p
    return None


def _load_rows() -> tuple[list[dict], list[str]]:
    """Return (all_rows, fieldnames) from CSV."""
    with open(CSV_PATH, encoding=CSV_ENCODING, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return rows, list(fieldnames)


def _save_rows(rows: list[dict], fieldnames: list[str]) -> None:
    with open(CSV_PATH, "w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── main app ─────────────────────────────────────────────────────────────────
class LabelApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("cut 재레이블링 툴")
        self.root.resizable(True, True)

        self.rows, self.fieldnames = _load_rows()

        # indices of rows where cut_to_label == UNLABELED_VALUE
        self.target_indices: list[int] = [
            i for i, r in enumerate(self.rows)
            if str(r.get("cut_to_label", "")).strip() == UNLABELED_VALUE
        ]
        self.total = len(self.target_indices)

        if self.total == 0:
            messagebox.showinfo("완료", "레이블할 이미지가 없습니다.")
            root.destroy()
            return

        # cursor into target_indices
        self.cursor = 0  # points to current item in target_indices
        # history for undo: list of (global_row_index, previous_value)
        self.history: list[tuple[int, str]] = []

        self._build_ui()
        self._show_current()

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        self.root.configure(bg="#1e1e1e")

        # progress label
        self.progress_var = tk.StringVar()
        progress_lbl = tk.Label(
            self.root,
            textvariable=self.progress_var,
            font=("Helvetica", 14, "bold"),
            fg="#cccccc",
            bg="#1e1e1e",
            pady=8,
        )
        progress_lbl.pack(fill="x")

        # fname label
        self.fname_var = tk.StringVar()
        fname_lbl = tk.Label(
            self.root,
            textvariable=self.fname_var,
            font=("Helvetica", 10),
            fg="#888888",
            bg="#1e1e1e",
            pady=2,
        )
        fname_lbl.pack(fill="x")

        # image canvas
        self.canvas = tk.Canvas(
            self.root,
            width=MAX_IMG_SIZE,
            height=MAX_IMG_SIZE,
            bg="#2d2d2d",
            highlightthickness=0,
        )
        self.canvas.pack(padx=16, pady=8)
        self._img_ref: ImageTk.PhotoImage | None = None

        # button row
        btn_frame = tk.Frame(self.root, bg="#1e1e1e", pady=12)
        btn_frame.pack()

        btn_cfg: list[tuple[str, str, str]] = [
            ("C  cut", "cut", "#3a7ab5"),
            ("U  UK", "UK", "#888800"),
            ("D  danger", "danger", "#b53a3a"),
            ("E  excluded", "excluded", "#3a9a5c"),
        ]
        self.buttons: dict[str, tk.Button] = {}
        for label_text, key, color in btn_cfg:
            btn = tk.Button(
                btn_frame,
                text=label_text,
                font=("Helvetica", 13, "bold"),
                width=11,
                bg=color,
                fg="white",
                activebackground=color,
                relief="flat",
                cursor="hand2",
                command=lambda k=key: self._label(k),
            )
            btn.pack(side="left", padx=6)
            self.buttons[key] = btn

        prev_btn = tk.Button(
            btn_frame,
            text="←  이전으로",
            font=("Helvetica", 13),
            width=11,
            bg="#444444",
            fg="white",
            activebackground="#555555",
            relief="flat",
            cursor="hand2",
            command=self._go_prev,
        )
        prev_btn.pack(side="left", padx=6)
        self.prev_btn = prev_btn

        # key bindings
        self.root.bind("<c>", lambda _: self._label("cut"))
        self.root.bind("<u>", lambda _: self._label("UK"))
        self.root.bind("<d>", lambda _: self._label("danger"))
        self.root.bind("<e>", lambda _: self._label("excluded"))
        self.root.bind("<Left>", lambda _: self._go_prev())

    # ── display ──────────────────────────────────────────────────────────────
    def _show_current(self) -> None:
        if self.cursor >= self.total:
            self._finish()
            return

        global_idx = self.target_indices[self.cursor]
        row = self.rows[global_idx]
        fname = row.get("fname", "")
        original_label = row.get("original_label", "")

        self.progress_var.set(
            f"{self.cursor + 1}  /  {self.total}   "
            f"(완료: {self.cursor}  /  남음: {self.total - self.cursor})"
        )
        self.fname_var.set(fname)

        img_path = _find_image(original_label, fname)
        self.canvas.delete("all")

        if img_path is None:
            self.canvas.create_text(
                MAX_IMG_SIZE // 2,
                MAX_IMG_SIZE // 2,
                text=f"이미지 없음\n{fname}",
                fill="#ff6666",
                font=("Helvetica", 14),
                justify="center",
            )
            # auto-label as UK
            self._apply_label("UK", global_idx, auto=True)
            return

        try:
            img = Image.open(img_path)
            img.thumbnail((MAX_IMG_SIZE, MAX_IMG_SIZE), Image.LANCZOS)
            self._img_ref = ImageTk.PhotoImage(img)
            w, h = img.size
            canvas_w = max(w, 400)
            canvas_h = max(h, 400)
            self.canvas.config(width=canvas_w, height=canvas_h)
            self.canvas.create_image(
                canvas_w // 2,
                canvas_h // 2,
                anchor="center",
                image=self._img_ref,
            )
        except Exception as exc:
            self.canvas.create_text(
                MAX_IMG_SIZE // 2,
                MAX_IMG_SIZE // 2,
                text=f"로드 실패: {exc}\n{fname}",
                fill="#ff6666",
                font=("Helvetica", 12),
                justify="center",
            )
            self._apply_label("UK", global_idx, auto=True)

    # ── actions ──────────────────────────────────────────────────────────────
    def _label(self, key: str) -> None:
        if self.cursor >= self.total:
            return
        global_idx = self.target_indices[self.cursor]
        self._apply_label(key, global_idx, auto=False)

    def _apply_label(self, key: str, global_idx: int, *, auto: bool) -> None:
        old_val = str(self.rows[global_idx].get("cut_to_label", "")).strip()
        new_val = str(LABEL_MAP[key])

        self.history.append((global_idx, old_val))
        self.rows[global_idx]["cut_to_label"] = new_val
        _save_rows(self.rows, self.fieldnames)

        self.cursor += 1

        if auto:
            # brief pause so user can see the auto-label message
            self.root.after(600, self._show_current)
        else:
            self._show_current()

    def _go_prev(self) -> None:
        if not self.history:
            return

        global_idx, prev_val = self.history.pop()

        # restore original value (UNLABELED_VALUE == "2")
        self.rows[global_idx]["cut_to_label"] = UNLABELED_VALUE
        _save_rows(self.rows, self.fieldnames)

        # move cursor back (find position of global_idx in target_indices)
        # since we always move forward, cursor-1 should be correct
        self.cursor = max(0, self.cursor - 1)
        self._show_current()

    def _finish(self) -> None:
        messagebox.showinfo("완료!", f"모든 {self.total}개 이미지 레이블 완료!")
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = LabelApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
