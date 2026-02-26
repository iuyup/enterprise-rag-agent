from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import fitz  # PyMuPDF


@dataclass
class LoadedDocument:
    source_path: str
    file_name: str
    file_type: str
    text: str
    meta: dict


def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def ensure_dirs() -> dict:
    base = Path("data")
    paths = {
        "uploads": base / "uploads",
        "processed": base / "processed",
        "index": base / "index",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return {k: str(v) for k, v in paths.items()}


def save_upload(file_name: str, file_bytes: bytes) -> str:
    """
    Save uploaded file to data/uploads with a stable unique name:
    <ts>_<sha1_8>_<orig_name>
    """
    dirs = ensure_dirs()
    ts = time.strftime("%Y%m%d_%H%M%S")
    h8 = _sha1_bytes(file_bytes)[:8]
    safe_name = file_name.replace("/", "_").replace("\\", "_")
    out_path = Path(dirs["uploads"]) / f"{ts}_{h8}_{safe_name}"
    out_path.write_bytes(file_bytes)
    return str(out_path)


def load_text_from_path(path: str) -> Tuple[str, dict]:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in [".txt", ".md", ".markdown"]:
        text = p.read_text(encoding="utf-8", errors="ignore")
        meta = {"loader": "text", "chars": len(text)}
        return text, meta

    if suffix == ".pdf":
        doc = fitz.open(path)
        parts = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            parts.append(page.get_text("text"))
        doc.close()
        text = "\n".join(parts).strip()
        meta = {"loader": "pymupdf", "pages": len(parts), "chars": len(text)}
        return text, meta

    raise ValueError(f"Unsupported file type: {suffix}")


def load_document(file_name: str, file_bytes: bytes) -> LoadedDocument:
    saved_path = save_upload(file_name, file_bytes)
    text, meta = load_text_from_path(saved_path)
    file_type = Path(saved_path).suffix.lower().lstrip(".")
    return LoadedDocument(
        source_path=saved_path,
        file_name=os.path.basename(saved_path),
        file_type=file_type,
        text=text,
        meta=meta,
    )