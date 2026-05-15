from __future__ import annotations

import argparse
import mmap
import re
import struct
import pathlib
import os
import datetime
from abc import ABC, abstractmethod
from typing import Any, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from context_logger import get_logger

log = get_logger("capture")

# Timezone shift string shared with bird_recog for directory hierarchy and filenames
_offset = datetime.datetime.now().astimezone().utcoffset()
_tzshift = int(((_offset.total_seconds() if _offset is not None else 0) / 60 / 60) * 100)
tzshift_str = f"{'p' if _tzshift >= 0 else 'm'}{abs(_tzshift):04d}"

MAGIC: int = 0xEFFEC51E
VERSION: int = 1
MAX_FILENAME_BYTES: int = 1024

# Superblock: magic(I) version(I) num_slots(I) write_head(I) max_image_bytes(I) index_entry_size(I) + 40 pad = 64 bytes
SUPERBLOCK_FORMAT = "<IIIIII40x"
SUPERBLOCK_SIZE = struct.calcsize(SUPERBLOCK_FORMAT)

# IndexEntry: image_offset(Q) image_size(Q) flags(Q) filename_len(I) reserved(I) filename(1024s) = 1056 bytes
INDEX_ENTRY_FORMAT = "<QQQII1024s"
INDEX_ENTRY_SIZE = struct.calcsize(INDEX_ENTRY_FORMAT)

TRIGGER_FRAME: int = 1


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--blob-capture-file",
        default="capture.blob",
        help="Name of the blob ring-buffer file, stored under the capture folder",
    )
    parser.add_argument(
        "--blob-num-slots",
        default=60,
        type=int,
        help="Number of image slots in the blob ring buffer",
    )
    parser.add_argument(
        "--blob-max-image-bytes",
        default=8 * 1024 * 1024,
        type=int,
        help="Maximum bytes per image slot in the blob ring buffer",
    )


class BlobCompletionHandler(ABC):
    """Invoked when a rotated event blob has received all its follow-up writes."""

    @abstractmethod
    def __call__(self, event_id: str, blob: BlobFsCapture) -> None: ...


class NoOpBlobCompletionHandler(BlobCompletionHandler):
    def __call__(self, event_id: str, blob: BlobFsCapture) -> None:
        pass


def _quality_to_compress_level(quality: int) -> int:
    return max(0, min(9, round((100 - quality) * 9 / 100)))


_PATTERN_RE = re.compile(r"X+")


def _resolve_blob_pattern(pattern: pathlib.Path) -> pathlib.Path:
    """Return the next sequential path for a pattern containing a run of 'X' chars.

    Globs the parent directory, finds the highest existing index, and returns a
    path with that index + 1 (starting at 1 when no files exist yet).
    Example: Path('/tmp/capture-XXXXX.blob') with 'capture-00001.blob' already
    present → Path('/tmp/capture-00002.blob').
    If the name contains no 'X' run, the path is returned unchanged.
    """
    match = _PATTERN_RE.search(pattern.name)
    if match is None:
        return pattern
    width = len(match.group())
    prefix = pattern.name[: match.start()]
    suffix = pattern.name[match.end() :]
    parent = pattern.parent

    existing = sorted(
        p for p in parent.glob(f"{prefix}{'?' * width}{suffix}") if p.name[len(prefix) : len(prefix) + width].isdigit()
    )
    next_idx = int(existing[-1].name[len(prefix) : len(prefix) + width]) + 1 if existing else 1
    return parent / f"{prefix}{next_idx:0{width}d}{suffix}"


class ImageCaptureInterface(ABC):
    @abstractmethod
    def capture(self, image: NDArray[np.uint8], filename: str, flags: int = 0) -> None: ...

    @abstractmethod
    def get_last_filepath(self) -> Optional[pathlib.Path]: ...

    @abstractmethod
    def set_capture_ts(self, ts: datetime.datetime) -> None: ...

    @abstractmethod
    def rotate(self, event_id: str, immediate: bool = False) -> ImageCaptureInterface: ...

    @abstractmethod
    def active_blob_path(self) -> Optional[pathlib.Path]: ...

    @abstractmethod
    def update_last_flags(self, flags: int) -> None: ...


class BlobFsCapture(ImageCaptureInterface):
    def __init__(
        self,
        blob_path: pathlib.Path,
        num_slots: int = 60,
        max_image_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self._blob_path = _resolve_blob_pattern(blob_path)
        self._num_slots = num_slots
        self._max_image_bytes = max_image_bytes
        self._index_base = SUPERBLOCK_SIZE
        self._data_base = SUPERBLOCK_SIZE + num_slots * INDEX_ENTRY_SIZE
        self._file_size = self._data_base + num_slots * max_image_bytes
        self._file: Optional[Any] = None
        self._mm: Optional[mmap.mmap] = None
        self._write_head = 0
        self._open()

    def _open(self) -> None:
        if not self._blob_path.exists():
            self._create()
        else:
            self._reopen()

    def _create(self) -> None:
        self._blob_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._blob_path, "wb") as f:
            f.seek(self._file_size - 1)
            f.write(b"\x00")
        self._file = open(self._blob_path, "r+b")
        self._mm = mmap.mmap(self._file.fileno(), self._file_size)
        self._write_head = 0
        self._flush_superblock()

    def _reopen(self) -> None:
        self._file = open(self._blob_path, "r+b")
        actual_size = os.path.getsize(self._blob_path)
        if actual_size != self._file_size:
            self._file.close()
            raise ValueError(f"Blob file size mismatch: expected {self._file_size}, got {actual_size}")
        self._mm = mmap.mmap(self._file.fileno(), self._file_size)
        fields = struct.unpack_from(SUPERBLOCK_FORMAT, self._mm, 0)
        magic = fields[0]
        if magic != MAGIC:
            self._mm.close()
            self._file.close()
            raise ValueError(f"Invalid blob magic: 0x{magic:08X}, expected 0x{MAGIC:08X}")
        self._write_head = fields[3]

    def _flush_superblock(self) -> None:
        data = struct.pack(
            SUPERBLOCK_FORMAT,
            MAGIC,
            VERSION,
            self._num_slots,
            self._write_head,
            self._max_image_bytes,
            INDEX_ENTRY_SIZE,
        )
        assert self._mm is not None
        self._mm[0:SUPERBLOCK_SIZE] = data

    def _slot_index_offset(self, slot: int) -> int:
        return self._index_base + slot * INDEX_ENTRY_SIZE

    def _slot_data_offset(self, slot: int) -> int:
        return self._data_base + slot * self._max_image_bytes

    def capture(self, image: NDArray[np.uint8], filename: str, flags: int = 0) -> None:
        assert self._mm is not None
        filename_bytes = filename.encode("utf-8")
        if len(filename_bytes) > MAX_FILENAME_BYTES:
            raise ValueError(f"Filename too long: {len(filename_bytes)} bytes, max {MAX_FILENAME_BYTES}")

        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("cv2.imencode failed to encode image as PNG")
        png_bytes = buf.tobytes()

        if len(png_bytes) > self._max_image_bytes:
            raise ValueError(f"PNG too large: {len(png_bytes)} bytes, max {self._max_image_bytes}")

        slot = self._write_head
        data_offset = self._slot_data_offset(slot)
        idx_offset = self._slot_index_offset(slot)

        # Write image data into the slot's region
        self._mm[data_offset : data_offset + len(png_bytes)] = png_bytes

        # Write index entry
        filename_padded = filename_bytes.ljust(MAX_FILENAME_BYTES, b"\x00")
        entry = struct.pack(
            INDEX_ENTRY_FORMAT,
            data_offset,
            len(png_bytes),
            flags,
            len(filename_bytes),
            0,
            filename_padded,
        )
        self._mm[idx_offset : idx_offset + INDEX_ENTRY_SIZE] = entry

        # Advance write_head and persist to superblock
        self._write_head = (self._write_head + 1) % self._num_slots
        struct.pack_into("<I", self._mm, 12, self._write_head)

        self._mm.flush()

    def get_last_filepath(self) -> Optional[pathlib.Path]:
        return None

    def set_capture_ts(self, ts: datetime.datetime) -> None:
        pass

    def active_blob_path(self) -> Optional[pathlib.Path]:
        return self._blob_path

    def update_last_flags(self, flags: int) -> None:
        assert self._mm is not None
        last_slot = (self._write_head - 1) % self._num_slots
        idx_offset = self._slot_index_offset(last_slot)
        struct.pack_into("<Q", self._mm, idx_offset + 16, flags)
        self._mm.flush()

    def rotate(self, event_id: str, immediate: bool = False) -> BlobFsCapture:
        """Rename the current blob to <event_id>.blob and open a fresh one at the original path.

        The current file handle and mmap are transferred to a new BlobFsCapture wrapper (the
        "old handle") which is returned to the caller.  On Linux, renaming an open/mmap'd file
        preserves the inode so the old handle remains valid after the rename.
        """
        old: BlobFsCapture = object.__new__(BlobFsCapture)
        old.__dict__.update(self.__dict__)

        renamed = self._blob_path.parent / f"{event_id}.blob"
        os.rename(self._blob_path, renamed)
        old._blob_path = renamed

        self._file = None
        self._mm = None
        self._write_head = 0
        self._open()

        return old

    def close(self) -> None:
        if self._mm is not None:
            self._mm.flush()
            self._mm.close()
            self._mm = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> BlobFsCapture:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class CompositeBlobCapture(ImageCaptureInterface):
    """Composite capture backend: one primary blob + a list of recently-rotated event blobs.

    After rotation each event blob receives `follow_up_count` additional frames (flags=0),
    then is closed and the completion handler is invoked.
    """

    def __init__(
        self,
        blob_path: pathlib.Path,
        num_slots: int,
        max_image_bytes: int,
        follow_up_count: int,
        completion_handler: BlobCompletionHandler,
    ) -> None:
        self._blob = BlobFsCapture(blob_path, num_slots, max_image_bytes)
        self._follow_up_count = follow_up_count
        self._completion_handler = completion_handler
        self._active: list[tuple[BlobFsCapture, str, int]] = []

    def set_capture_ts(self, ts: datetime.datetime) -> None:
        self._blob.set_capture_ts(ts)

    def get_last_filepath(self) -> Optional[pathlib.Path]:
        return self._blob.get_last_filepath()

    def capture(self, image: NDArray[np.uint8], filename: str, flags: int = 0) -> None:
        self._blob.capture(image, filename, flags)
        next_active: list[tuple[BlobFsCapture, str, int]] = []
        for blob, event_id, remaining in self._active:
            blob.capture(image, filename, 0)
            remaining -= 1
            if remaining > 0:
                next_active.append((blob, event_id, remaining))
            else:
                blob.close()
                self._completion_handler(event_id, blob)
        self._active = next_active

    def rotate(self, event_id: str, immediate: bool = False) -> CompositeBlobCapture:
        old = self._blob.rotate(event_id)
        if immediate:
            old.close()
            self._completion_handler(event_id, old)
        else:
            self._active.append((old, event_id, self._follow_up_count))
        return self

    def close(self) -> None:
        for blob, _event_id, _remaining in self._active:
            blob.close()
        self._active = []
        self._blob.close()

    def active_blob_path(self) -> Optional[pathlib.Path]:
        return self._blob._blob_path

    def update_last_flags(self, flags: int) -> None:
        self._blob.update_last_flags(flags)


class BlobExtractor:
    def __init__(self, blob_path: pathlib.Path) -> None:
        self._blob_path = blob_path

    def extract(self, dest_dir: pathlib.Path) -> list[pathlib.Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        data = self._blob_path.read_bytes()

        if len(data) < SUPERBLOCK_SIZE:
            raise ValueError("File too small to contain a valid superblock")

        magic, _version, num_slots, _write_head, _max_image_bytes, index_entry_size = struct.unpack_from(
            SUPERBLOCK_FORMAT, data, 0
        )
        if magic != MAGIC:
            raise ValueError(f"Invalid blob magic: 0x{magic:08X}, expected 0x{MAGIC:08X}")

        extracted: list[pathlib.Path] = []
        for i in range(num_slots):
            idx_offset = SUPERBLOCK_SIZE + i * index_entry_size
            if idx_offset + index_entry_size > len(data):
                break

            image_offset, image_size, _flags, filename_len, _reserved, filename_raw = struct.unpack_from(
                INDEX_ENTRY_FORMAT, data, idx_offset
            )

            if image_size == 0:
                continue
            if image_offset + image_size > len(data):
                continue

            png_bytes = data[image_offset : image_offset + image_size]
            arr = np.frombuffer(png_bytes, dtype=np.uint8)
            if cv2.imdecode(arr, cv2.IMREAD_COLOR) is None:
                continue

            raw_fn = filename_raw[:filename_len]
            try:
                filename = raw_fn.decode("utf-8")
            except UnicodeDecodeError:
                filename = raw_fn.decode("latin-1")

            if not filename:
                filename = f"slot_{i:04d}.png"

            dest_path = dest_dir / filename
            dest_path.write_bytes(png_bytes)
            extracted.append(dest_path)

        return extracted


def main() -> None:
    import argparse as _argparse
    import sys

    parser = _argparse.ArgumentParser(
        prog="er-scarecrow-blob-extract",
        description="Extract PNG images from a BlobFsCapture ring-buffer file.",
        formatter_class=_argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("blob", type=pathlib.Path, help="Path to the .blob file")
    parser.add_argument(
        "dest",
        type=pathlib.Path,
        nargs="?",
        default=pathlib.Path("."),
        help="Destination directory for extracted images (created if absent)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print blob metadata and slot summary without extracting images",
    )
    args = parser.parse_args()

    if not args.blob.exists():
        print(f"error: blob file not found: {args.blob}", file=sys.stderr)
        sys.exit(1)

    data = args.blob.read_bytes()
    if len(data) < SUPERBLOCK_SIZE:
        print("error: file too small to be a valid blob", file=sys.stderr)
        sys.exit(1)

    magic, version, num_slots, write_head, max_image_bytes, index_entry_size = struct.unpack_from(
        SUPERBLOCK_FORMAT, data, 0
    )
    if magic != MAGIC:
        print(f"error: invalid magic 0x{magic:08X} (expected 0x{MAGIC:08X})", file=sys.stderr)
        sys.exit(1)

    trigger_names = get_trigger_names_from_blob(data, num_slots, index_entry_size)

    if args.info:
        print_blob_info(args, data, version, num_slots, write_head, max_image_bytes, index_entry_size, trigger_names)
        return

    extracted = BlobExtractor(args.blob).extract(args.dest)
    for path in sorted(extracted):
        marker = "  [TRIGGER_FRAME]" if path.name in trigger_names else ""
        print(f"{path}{marker}")
    print(f"\nextracted {len(extracted)} image(s) to {args.dest}", file=sys.stderr)


def print_blob_info(
    args: Any,
    data: bytes,
    version: int,
    num_slots: int,
    write_head: int,
    max_image_bytes: int,
    index_entry_size: int,
    trigger_names: set[str],
) -> None:
    occupied = sum(
        1 for i in range(num_slots) if struct.unpack_from("<Q", data, SUPERBLOCK_SIZE + i * index_entry_size + 8)[0] > 0
    )

    def _fmt(n: int) -> str:
        if n >= 1024 * 1024:
            return f"{n / 1024 / 1024:.1f} MiB ({n} bytes)"
        return f"{n / 1024:.1f} KiB ({n} bytes)"

    print(f"blob:             {args.blob}")
    print(f"magic:            {MAGIC:#010x}")
    print(f"version:          {version}")
    print(f"num_slots:        {num_slots}  (write_head={write_head})")
    print(f"max_image_bytes:  {_fmt(max_image_bytes)}")
    print(f"index_entry_size: {index_entry_size}")
    print(f"file_size:        {_fmt(len(data))}")
    print(f"occupied_slots:   {occupied}/{num_slots}")
    for trigger in sorted(trigger_names):
        print(f"trigger_frame: {trigger}")


def get_trigger_names_from_blob(data: bytes, num_slots: int, index_entry_size: int) -> set[str]:
    trigger_names: set[str] = set()
    for i in range(num_slots):
        idx_offset = SUPERBLOCK_SIZE + i * index_entry_size
        if idx_offset + index_entry_size > len(data):
            break
        _img_off, img_size, flags, fn_len, _res, fn_raw = struct.unpack_from(INDEX_ENTRY_FORMAT, data, idx_offset)
        if img_size > 0 and flags & TRIGGER_FRAME:
            raw_fn = fn_raw[:fn_len]
            try:
                fn = raw_fn.decode("utf-8")
            except UnicodeDecodeError:
                fn = raw_fn.decode("latin-1")
            trigger_names.add(fn)
    return trigger_names


def make_capture_backend(
    args: argparse.Namespace,
    capture_folder: pathlib.Path,
    follow_up_count: int,
    completion_handler: Optional[BlobCompletionHandler] = None,
) -> ImageCaptureInterface:
    blob_path = capture_folder / args.blob_capture_file
    num_slots: int = args.blob_num_slots
    max_image_bytes: int = args.blob_max_image_bytes
    handler = completion_handler if completion_handler is not None else NoOpBlobCompletionHandler()
    return CompositeBlobCapture(blob_path, num_slots, max_image_bytes, follow_up_count, handler)
