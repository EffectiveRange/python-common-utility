from __future__ import annotations

import argparse
import dataclasses
import mmap
import shutil
import struct
import pathlib
import os
import sys
import datetime
import tarfile
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

MAGIC: int = 0xEFFEC51E
VERSION: int = 1
MAX_FILENAME_BYTES: int = 1024

# Superblock: magic(I) version(I) num_slots(I) write_head(I) max_image_bytes(I) index_entry_size(I) + 40 pad = 64 bytes
SUPERBLOCK_FORMAT = "<IIIIII40x"
SUPERBLOCK_SIZE = struct.calcsize(SUPERBLOCK_FORMAT)

# IndexEntry: image_offset(Q) image_size(Q) flags(Q) filename_len(I) reserved(I) filename(1024s) = 1056 bytes
INDEX_ENTRY_FORMAT = "<QQQII1024s"
INDEX_ENTRY_SIZE = struct.calcsize(INDEX_ENTRY_FORMAT)

TRIGGER_FRAME_FLAG: int = 1
RAW_IMAGE_FLAG: int = 2

RAW_IMAGE_HEADER_FORMAT = "<III"  # height, width, channels
RAW_IMAGE_HEADER_SIZE = struct.calcsize(RAW_IMAGE_HEADER_FORMAT)


def add_args(parser: argparse.ArgumentParser, **defaults: Any) -> None:
    parser.add_argument(
        "--blob-capture-file",
        default=pathlib.Path(defaults.get("blob_capture_file", "capture.blob")),
        help="Full path to the active blob ring-buffer file (e.g. a tmpfs location)",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--blob-capture-folder",
        default=pathlib.Path(defaults.get("blob_capture_folder", ".")),
        help="Folder for the rotated blob capture files (created if absent)",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--blob-num-slots",
        default=int(defaults.get("blob_num_slots", 60)),
        type=int,
        help="Number of image slots in the blob ring buffer",
    )
    parser.add_argument(
        "--blob-max-image-bytes",
        default=int(defaults.get("blob_max_image_bytes", 4 * 1024 * 1024)),
        type=int,
        help="Maximum bytes per image slot in the blob ring buffer",
    )
    parser.add_argument(
        "--blob-png-compression",
        default=int(defaults.get("blob_png_compression", 1)),
        type=int,
        choices=range(10),
        metavar="0-9",
        help="PNG compression level for captured images (0=none, 9=max); default 1 favours throughput",
    )
    parser.add_argument(
        "--blob-raw-capture",
        action="store_true",
        default=bool(defaults.get("blob_raw_capture", False)),
        help="Store images as raw pixel arrays instead of PNG (faster capture, decoded to PNG on extraction)",
    )


class BlobCompletionHandler(ABC):
    """Invoked when a rotation event is complete.

    blob_path is the primary (pre-rotation) blob; additional_blobs contains the post blob
    path when one was opened (empty for immediate rotations).
    """

    @abstractmethod
    def __call__(
        self,
        event_id: str,
        blob_path: pathlib.Path,
        additional_blobs: list[pathlib.Path],
    ) -> None: ...


class NoOpBlobCompletionHandler(BlobCompletionHandler):
    def __call__(
        self,
        event_id: str,
        blob_path: pathlib.Path,
        additional_blobs: list[pathlib.Path],
    ) -> None:
        pass


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
        max_image_bytes: int = 4 * 1024 * 1024,
        png_compression: int = 1,
        capture_folder: Optional[pathlib.Path] = None,
        raw_capture: bool = False,
    ) -> None:
        self._blob_path = blob_path
        self._num_slots = num_slots
        self._max_image_bytes = max_image_bytes
        self._png_compression = png_compression
        self._capture_folder = capture_folder
        self._raw_capture = raw_capture
        self._index_base = SUPERBLOCK_SIZE
        self._data_base = SUPERBLOCK_SIZE + num_slots * INDEX_ENTRY_SIZE
        self._file_size = self._data_base + num_slots * max_image_bytes
        self._file: Optional[Any] = None
        self._mm: Optional[mmap.mmap] = None
        self._write_head = 0
        self._open()

    def _open(self) -> None:
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
        if self._blob_path.exists():
            os.unlink(self._blob_path)
        self._create()

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

        if self._raw_capture:
            h, w = image.shape[:2]
            c = image.shape[2] if image.ndim == 3 else 1
            header = struct.pack(RAW_IMAGE_HEADER_FORMAT, h, w, c)
            image_size = RAW_IMAGE_HEADER_SIZE + image.nbytes
            actual_flags = flags | RAW_IMAGE_FLAG
        else:
            ok, buf = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, self._png_compression])
            if not ok:
                raise ValueError("cv2.imencode failed to encode image as PNG")
            image_size = len(buf)
            actual_flags = flags

        if image_size > self._max_image_bytes:
            raise ValueError(f"Image data too large: {image_size} bytes, max {self._max_image_bytes}")

        slot = self._write_head
        data_offset = self._slot_data_offset(slot)
        idx_offset = self._slot_index_offset(slot)

        if self._raw_capture:
            self._mm[data_offset : data_offset + RAW_IMAGE_HEADER_SIZE] = header
            self._mm[data_offset + RAW_IMAGE_HEADER_SIZE : data_offset + image_size] = image
        else:
            self._mm[data_offset : data_offset + image_size] = buf

        filename_padded = filename_bytes.ljust(MAX_FILENAME_BYTES, b"\x00")
        entry = struct.pack(
            INDEX_ENTRY_FORMAT,
            data_offset,
            image_size,
            actual_flags,
            len(filename_bytes),
            0,
            filename_padded,
        )
        self._mm[idx_offset : idx_offset + INDEX_ENTRY_SIZE] = entry

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
        current = struct.unpack_from("<Q", self._mm, idx_offset + 16)[0]
        struct.pack_into("<Q", self._mm, idx_offset + 16, current | flags)
        self._mm.flush()

    def rotate(self, event_id: str, immediate: bool = False) -> BlobFsCapture:
        """Close the current blob, move it to capture_folder (or same dir) as <event_id>.blob, then reopen fresh."""
        self.close()

        if self._capture_folder is not None:
            self._capture_folder.mkdir(parents=True, exist_ok=True)
            dest = self._capture_folder / f"{event_id}.blob"
            shutil.move(str(self._blob_path), str(dest))
        else:
            dest = self._blob_path.parent / f"{event_id}.blob"
            os.rename(self._blob_path, dest)

        self._open()
        return self

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
    """Composite capture backend: one primary blob on a fast path (e.g. tmpfs) + one optional
    post-event blob in the capture folder receiving follow-up frames after rotation.

    On rotate(event_id):
      - The primary blob is closed and moved to capture_folder/{event_id}.blob.
      - A fresh primary blob is opened at the original blob_path.
      - A new {event_id}-post.blob is opened in capture_folder to receive follow_up_count frames.
      - At most one post blob is active at a time; a second rotation force-completes the existing one.
    """

    def __init__(
        self,
        blob_path: pathlib.Path,
        capture_folder: pathlib.Path,
        num_slots: int,
        max_image_bytes: int,
        follow_up_count: int,
        completion_handler: BlobCompletionHandler,
        png_compression: int = 1,
        raw_capture: bool = False,
    ) -> None:
        self._capture_folder = capture_folder
        self._num_slots = num_slots
        self._max_image_bytes = max_image_bytes
        self._png_compression = png_compression
        self._raw_capture = raw_capture
        self._blob = BlobFsCapture(
            blob_path,
            num_slots,
            max_image_bytes,
            png_compression,
            capture_folder=capture_folder,
            raw_capture=raw_capture,
        )
        self._follow_up_count = follow_up_count
        self._completion_handler = completion_handler
        self._post_blob: Optional[tuple[BlobFsCapture, str, int]] = None

    def set_capture_ts(self, ts: datetime.datetime) -> None:
        self._blob.set_capture_ts(ts)

    def get_last_filepath(self) -> Optional[pathlib.Path]:
        return self._blob.get_last_filepath()

    def capture(self, image: NDArray[np.uint8], filename: str, flags: int = 0) -> None:
        self._blob.capture(image, filename, flags)
        if self._post_blob is not None:
            post_b, post_event_id, remaining = self._post_blob
            post_b.capture(image, filename, 0)
            remaining -= 1
            if remaining > 0:
                self._post_blob = (post_b, post_event_id, remaining)
            else:
                post_b.close()
                self._completion_handler(
                    post_event_id,
                    self._capture_folder / f"{post_event_id}.blob",
                    [self._capture_folder / f"{post_event_id}-post.blob"],
                )
                self._post_blob = None

    def rotate(self, event_id: str, immediate: bool = False) -> CompositeBlobCapture:
        # Force-complete any active post blob before starting a new rotation
        if self._post_blob is not None:
            post_b, post_event_id, _ = self._post_blob
            post_b.close()
            self._completion_handler(
                post_event_id,
                self._capture_folder / f"{post_event_id}.blob",
                [self._capture_folder / f"{post_event_id}-post.blob"],
            )
            self._post_blob = None

        self._blob.rotate(event_id)

        if immediate:
            self._completion_handler(event_id, self._capture_folder / f"{event_id}.blob", [])
        else:
            post_path = self._capture_folder / f"{event_id}-post.blob"
            post_blob = BlobFsCapture(
                post_path, self._num_slots, self._max_image_bytes, self._png_compression, raw_capture=self._raw_capture
            )
            self._post_blob = (post_blob, event_id, self._follow_up_count)

        return self

    def close(self) -> None:
        if self._post_blob is not None:
            self._post_blob[0].close()
            self._post_blob = None
        self._blob.close()

    def active_blob_path(self) -> Optional[pathlib.Path]:
        return self._blob._blob_path

    def update_last_flags(self, flags: int) -> None:
        self._blob.update_last_flags(flags)


@dataclasses.dataclass
class _OpenBlob:
    file: Optional[Any]
    mm: Optional[mmap.mmap]
    version: int
    num_slots: int
    write_head: int
    max_image_bytes: int
    index_entry_size: int


class BlobExtractor:
    def __init__(self, blob_path: pathlib.Path) -> None:
        self._blob_path = blob_path
        self._blobs: list[_OpenBlob] = []
        self._tempdir: Optional[tempfile.TemporaryDirectory] = None  # type: ignore[type-arg]
        self._frame_flags: Optional[dict[str, int]] = None
        try:
            paths = self._resolve_blob_paths(blob_path)
            for path in paths:
                self._open_single_blob(path)
        except Exception:
            self.close()
            raise

        # Backwards-compat public attributes pointing to the first blob
        first = self._blobs[0]
        self._file = first.file
        self._mm = first.mm
        self.version = first.version
        self.num_slots = first.num_slots
        self.write_head = first.write_head
        self.max_image_bytes = first.max_image_bytes
        self.index_entry_size = first.index_entry_size

    def _open_single_blob(self, path: pathlib.Path) -> None:
        file = open(path, "rb")
        try:
            file_size = os.path.getsize(path)
            if file_size < SUPERBLOCK_SIZE:
                raise ValueError("File too small to contain a valid superblock")
            mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            magic, version, num_slots, write_head, max_image_bytes, index_entry_size = struct.unpack_from(
                SUPERBLOCK_FORMAT, mm, 0
            )
            if magic != MAGIC:
                raise ValueError(f"Invalid blob magic: 0x{magic:08X}, expected 0x{MAGIC:08X}")
        except Exception:
            file.close()
            raise
        self._blobs.append(_OpenBlob(file, mm, version, num_slots, write_head, max_image_bytes, index_entry_size))

    def _resolve_blob_paths(self, blob_path: pathlib.Path) -> list[pathlib.Path]:
        if tarfile.is_tarfile(blob_path):
            return self._extract_tar_blobs(blob_path)
        # Regular file: include the primary plus any {stem}-*.blob siblings
        parent = blob_path.parent
        stem = blob_path.stem
        siblings = sorted(parent.glob(f"{stem}-*.blob"))
        return [blob_path] + siblings

    def _extract_tar_blobs(self, blob_path: pathlib.Path) -> list[pathlib.Path]:
        tmpdir = tempfile.TemporaryDirectory()
        self._tempdir = tmpdir
        try:
            paths: list[pathlib.Path] = []
            with tarfile.open(blob_path) as tf:
                members = [m for m in tf.getmembers() if m.isfile()]
                blob_members = [m for m in members if m.name.endswith(".blob")]
                targets = blob_members if blob_members else (members[:1] if members else [])
                if not targets:
                    raise ValueError(f"No regular file found in archive: {blob_path}")
                for member in targets:
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        continue
                    actual_path = pathlib.Path(tmpdir.name) / pathlib.Path(member.name).name
                    actual_path.write_bytes(extracted.read())
                    paths.append(actual_path)
            if not paths:
                raise ValueError(f"No regular file found in archive: {blob_path}")
            return paths
        except Exception:
            tmpdir.cleanup()
            self._tempdir = None
            raise

    def close(self) -> None:
        for b in self._blobs:
            if b.mm is not None:
                b.mm.close()
                b.mm = None
            if b.file is not None:
                b.file.close()
                b.file = None
        self._blobs = []
        self._mm = None
        self._file = None
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def __enter__(self) -> BlobExtractor:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _extract_png(self, raw_data: bytes) -> Optional[bytes]:
        arr = np.frombuffer(raw_data, dtype=np.uint8)
        if cv2.imdecode(arr, cv2.IMREAD_COLOR) is None:
            return None
        return raw_data

    def _extract_raw(self, raw_data: bytes) -> Optional[bytes]:
        if len(raw_data) < RAW_IMAGE_HEADER_SIZE:
            return None
        h, w, c = struct.unpack_from(RAW_IMAGE_HEADER_FORMAT, raw_data, 0)
        pixel_data = raw_data[RAW_IMAGE_HEADER_SIZE:]
        if len(pixel_data) != h * w * c:
            return None
        arr = np.frombuffer(pixel_data, dtype=np.uint8).reshape(h, w, c)
        ok, buf = cv2.imencode(".png", arr)
        if not ok:
            return None
        return buf.tobytes()

    def _extract_one(self, i: int, blob: _OpenBlob, dest_dir: pathlib.Path) -> Optional[pathlib.Path]:
        idx_offset = SUPERBLOCK_SIZE + i * blob.index_entry_size
        assert blob.mm is not None
        if idx_offset + blob.index_entry_size > len(blob.mm):
            return None

        image_offset, image_size, _flags, filename_len, _reserved, filename_raw = struct.unpack_from(
            INDEX_ENTRY_FORMAT, blob.mm, idx_offset
        )

        if image_size == 0 or image_offset + image_size > len(blob.mm):
            return None

        raw_data = bytes(blob.mm[image_offset : image_offset + image_size])

        if _flags & RAW_IMAGE_FLAG:
            png_bytes = self._extract_raw(raw_data)
        else:
            png_bytes = self._extract_png(raw_data)

        if png_bytes is None:
            return None

        raw_fn = filename_raw[:filename_len]
        try:
            filename = raw_fn.decode("utf-8")
        except UnicodeDecodeError:
            filename = raw_fn.decode("latin-1")

        if not filename:
            filename = f"slot_{i:04d}.png"

        dest_path = dest_dir / str(filename)
        dest_path.write_bytes(png_bytes)
        return dest_path

    def _extract_blob(self, blob: _OpenBlob, dest_dir: pathlib.Path) -> list[pathlib.Path]:
        assert blob.mm is not None
        extracted: list[pathlib.Path] = []
        for i in range(blob.num_slots):
            dst = self._extract_one(i, blob, dest_dir)
            if dst is None:
                continue
            extracted.append(dst)

        return extracted

    def extract(self, dest_dir: pathlib.Path) -> list[pathlib.Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        extracted: list[pathlib.Path] = []
        for blob in self._blobs:
            extracted.extend(self._extract_blob(blob, dest_dir))
        return extracted

    def _flags_for_blob(self, blob: _OpenBlob, mask: int) -> set[str]:
        assert blob.mm is not None
        result: set[str] = set()
        for i in range(blob.num_slots):
            idx_offset = SUPERBLOCK_SIZE + i * blob.index_entry_size
            if idx_offset + blob.index_entry_size > len(blob.mm):
                break
            _, img_size, flags, fn_len, _, fn_raw = struct.unpack_from(INDEX_ENTRY_FORMAT, blob.mm, idx_offset)
            if img_size > 0 and flags & mask:
                raw_fn = fn_raw[:fn_len]
                try:
                    fn = raw_fn.decode("utf-8")
                except UnicodeDecodeError:
                    fn = raw_fn.decode("latin-1")
                result.add(fn)
        return result

    def get_filenames_with_flags(self, mask: int) -> set[str]:
        result: set[str] = set()
        for blob in self._blobs:
            result |= self._flags_for_blob(blob, mask)
        return result


def main() -> None:

    parser = argparse.ArgumentParser(
        prog="er-blob-extract",
        description="Extract PNG images from a BlobFsCapture ring-buffer file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dest",
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
    parser.add_argument("blob", type=pathlib.Path, help="Path to the blob file or tar archive to extract")
    args = parser.parse_args()

    try:
        extractor = BlobExtractor(args.blob)
    except FileNotFoundError:
        print(f"error: blob file not found: {args.blob}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.info:
        assert extractor._mm is not None
        print_blob_info(
            args,
            extractor._mm,
            extractor.version,
            extractor.num_slots,
            extractor.write_head,
            extractor.max_image_bytes,
            extractor.index_entry_size,
            extractor.get_filenames_with_flags(TRIGGER_FRAME_FLAG),
        )
        return

    extracted = extractor.extract(args.dest)
    for path in sorted(extracted):
        marker = "  [TRIGGER_FRAME_FLAG]" if path.name in extractor.get_filenames_with_flags(TRIGGER_FRAME_FLAG) else ""
        print(f"{path}{marker}")
    print(f"\nextracted {len(extracted)} image(s) to {args.dest}", file=sys.stderr)


def print_blob_info(
    args: Any,
    data: bytes | mmap.mmap,
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


def make_capture_backend(
    args: argparse.Namespace,
    follow_up_count: int,
    completion_handler: Optional[BlobCompletionHandler] = None,
) -> ImageCaptureInterface:
    blob_path: pathlib.Path = args.blob_capture_file
    capture_folder: pathlib.Path = args.blob_capture_folder
    num_slots: int = args.blob_num_slots
    max_image_bytes: int = args.blob_max_image_bytes
    png_compression: int = args.blob_png_compression
    raw_capture: bool = args.blob_raw_capture
    handler = completion_handler if completion_handler is not None else NoOpBlobCompletionHandler()
    return CompositeBlobCapture(
        blob_path, capture_folder, num_slots, max_image_bytes, follow_up_count, handler, png_compression, raw_capture
    )
