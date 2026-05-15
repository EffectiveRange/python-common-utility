import argparse
import datetime
import io
import pathlib
import struct
import sys

import numpy as np
import pytest

from unittest.mock import MagicMock, patch

from common_utility.capture import (
    MAGIC,
    SUPERBLOCK_FORMAT,
    SUPERBLOCK_SIZE,
    INDEX_ENTRY_FORMAT,
    INDEX_ENTRY_SIZE,
    TRIGGER_FRAME_FLAG,
    BlobCompletionHandler,
    NoOpBlobCompletionHandler,
    CompositeBlobCapture,
    BlobFsCapture,
    BlobExtractor,
    MAX_FILENAME_BYTES,
    add_args,
    make_capture_backend,
    main as blob_extract_main,
)


def _make_image(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# BlobFsCapture
# ---------------------------------------------------------------------------


class BlobFsCaptureTest:
    def test_creates_blob_file_with_correct_size(self, tmp_path: pathlib.Path) -> None:
        num_slots = 4
        max_bytes = 512 * 1024
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=num_slots, max_image_bytes=max_bytes)
        blob.close()
        expected = SUPERBLOCK_SIZE + num_slots * INDEX_ENTRY_SIZE + num_slots * max_bytes
        assert (tmp_path / "cap.blob").stat().st_size == expected

    def test_superblock_magic_on_new_file(self, tmp_path: pathlib.Path) -> None:
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=4, max_image_bytes=256 * 1024)
        blob.close()
        data = (tmp_path / "cap.blob").read_bytes()
        magic = struct.unpack_from("<I", data, 0)[0]
        assert magic == MAGIC

    def test_reopen_validates_magic(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "cap.blob"
        blob = BlobFsCapture(path, num_slots=4, max_image_bytes=256 * 1024)
        blob.close()
        # Corrupt magic
        raw = bytearray(path.read_bytes())
        struct.pack_into("<I", raw, 0, 0xDEADBEEF)
        path.write_bytes(bytes(raw))
        with pytest.raises(ValueError, match="magic"):
            BlobFsCapture(path, num_slots=4, max_image_bytes=256 * 1024)

    def test_write_and_ring_wrap(self, tmp_path: pathlib.Path) -> None:
        num_slots = 3
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=num_slots, max_image_bytes=512 * 1024)
        images = [_make_image() for _ in range(num_slots + 1)]
        filenames = [f"frame_{i}.png" for i in range(num_slots + 1)]
        for img, fn in zip(images, filenames):
            blob.capture(img, fn)
        blob.close()

        # write_head should have wrapped: after 4 writes into 3 slots, write_head == 1
        data = (tmp_path / "cap.blob").read_bytes()
        _, _, _, write_head, _, _ = struct.unpack_from(SUPERBLOCK_FORMAT, data, 0)
        assert write_head == 1

        # Slot 0 should contain the 4th image (index 3)
        idx0_offset = SUPERBLOCK_SIZE
        img_offset, img_size, _flags, fn_len, _res, fn_raw = struct.unpack_from(INDEX_ENTRY_FORMAT, data, idx0_offset)
        fn = fn_raw[:fn_len].decode("utf-8")
        assert fn == "frame_3.png"

    def test_filename_too_long_raises(self, tmp_path: pathlib.Path) -> None:
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=4, max_image_bytes=256 * 1024)
        with pytest.raises(ValueError, match="too long"):
            blob.capture(_make_image(), "x" * (MAX_FILENAME_BYTES + 1))
        blob.close()

    def test_image_too_large_raises(self, tmp_path: pathlib.Path) -> None:
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=4, max_image_bytes=1)
        with pytest.raises(ValueError, match="too large"):
            blob.capture(_make_image(), "frame.png")
        blob.close()

    def test_context_manager(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "cap.blob"
        with BlobFsCapture(path, num_slots=4, max_image_bytes=256 * 1024) as blob:
            blob.capture(_make_image(), "frame.png")
        assert path.exists()

    def test_flags_stored_in_index(self, tmp_path: pathlib.Path) -> None:
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=4, max_image_bytes=512 * 1024)
        blob.capture(_make_image(), "frame.png", flags=0xCAFEBABE)
        blob.close()
        data = (tmp_path / "cap.blob").read_bytes()
        _img_off, _img_sz, flags, _fn_len, _res, _fn_raw = struct.unpack_from(INDEX_ENTRY_FORMAT, data, SUPERBLOCK_SIZE)
        assert flags == 0xCAFEBABE

    def test_update_last_flags_overwrites_flags_field(self, tmp_path: pathlib.Path) -> None:
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=4, max_image_bytes=512 * 1024)
        blob.capture(_make_image(), "frame.png", flags=0)
        blob.update_last_flags(TRIGGER_FRAME_FLAG)
        blob.close()
        data = (tmp_path / "cap.blob").read_bytes()
        _img_off, _img_sz, flags, _fn_len, _res, _fn_raw = struct.unpack_from(INDEX_ENTRY_FORMAT, data, SUPERBLOCK_SIZE)
        assert flags == TRIGGER_FRAME_FLAG

    def test_update_last_flags_targets_most_recent_slot(self, tmp_path: pathlib.Path) -> None:
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=4, max_image_bytes=512 * 1024)
        blob.capture(_make_image(), "first.png", flags=0)
        blob.capture(_make_image(), "second.png", flags=0)
        blob.update_last_flags(TRIGGER_FRAME_FLAG)
        blob.close()
        data = (tmp_path / "cap.blob").read_bytes()
        # slot 0 (first) must stay 0
        _a, _b, flags0, *_ = struct.unpack_from(INDEX_ENTRY_FORMAT, data, SUPERBLOCK_SIZE)
        assert flags0 == 0
        # slot 1 (second/last) must have TRIGGER_FRAME_FLAG
        _a, _b, flags1, *_ = struct.unpack_from(INDEX_ENTRY_FORMAT, data, SUPERBLOCK_SIZE + INDEX_ENTRY_SIZE)
        assert flags1 == TRIGGER_FRAME_FLAG


# ---------------------------------------------------------------------------
# BlobExtractor
# ---------------------------------------------------------------------------


class BlobExtractorTest:
    def _write_blob(self, tmp_path: pathlib.Path, images_and_names: list[tuple]) -> pathlib.Path:
        path = tmp_path / "cap.blob"
        blob = BlobFsCapture(path, num_slots=5, max_image_bytes=512 * 1024)
        for img, fn in images_and_names:
            blob.capture(img, fn)
        blob.close()
        return path

    def test_extract_valid_images(self, tmp_path: pathlib.Path) -> None:
        names = ["a.png", "b.png", "c.png"]
        path = self._write_blob(tmp_path, [(_make_image(), n) for n in names])
        dest = tmp_path / "out"
        extracted = BlobExtractor(path).extract(dest)
        assert len(extracted) == 3
        assert {p.name for p in extracted} == set(names)
        for p in extracted:
            assert p.exists()

    def test_partial_blob_is_decodable(self, tmp_path: pathlib.Path) -> None:
        """A blob with fewer written entries than its slot capacity must still decode correctly."""
        num_slots = 5
        written_names = ["frame_0.png", "frame_1.png", "frame_2.png"]
        path = tmp_path / "partial.blob"

        blob = BlobFsCapture(path, num_slots=num_slots, max_image_bytes=512 * 1024)
        for name in written_names:
            blob.capture(_make_image(), name)
        blob.close()

        extracted = BlobExtractor(path).extract(tmp_path / "out")
        assert len(extracted) == len(written_names)
        assert {p.name for p in extracted} == set(written_names)

    def test_extract_creates_dest_dir(self, tmp_path: pathlib.Path) -> None:
        path = self._write_blob(tmp_path, [(_make_image(), "a.png")])
        dest = tmp_path / "new" / "subdir"
        BlobExtractor(path).extract(dest)
        assert dest.exists()

    def test_extract_invalid_magic_raises(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "bad.blob"
        path.write_bytes(b"\xff" * 64)
        with pytest.raises(ValueError, match="magic"):
            BlobExtractor(path).extract(tmp_path / "out")

    def test_corruption_resilience(self, tmp_path: pathlib.Path) -> None:
        """Simulates two interrupted writes; BlobExtractor must extract the 3 valid images."""
        path = tmp_path / "cap.blob"
        blob = BlobFsCapture(path, num_slots=5, max_image_bytes=512 * 1024)
        for i in range(3):
            blob.capture(_make_image(), f"valid_{i}.png")
        blob.close()

        raw = bytearray(path.read_bytes())

        # Scenario A: slot 3 — advance write_head in superblock but leave index entry zeroed
        struct.pack_into("<I", raw, 12, 4)  # write_head = 4 (as if 4 writes happened)
        # Index entry for slot 3 stays all-zeros (image_size == 0)

        # Scenario B: slot 4 — write garbage bytes as image data, set a non-zero image_size
        sb_fields = struct.unpack_from(SUPERBLOCK_FORMAT, raw, 0)
        num_slots = sb_fields[2]
        max_image_bytes = sb_fields[4]
        slot4_data_offset = SUPERBLOCK_SIZE + num_slots * INDEX_ENTRY_SIZE + 4 * max_image_bytes
        slot4_idx_offset = SUPERBLOCK_SIZE + 4 * INDEX_ENTRY_SIZE
        garbage = b"\xde\xad\xbe\xef" * 16
        raw[slot4_data_offset : slot4_data_offset + len(garbage)] = garbage
        fn = b"corrupt.png".ljust(1024, b"\x00")
        entry = struct.pack(INDEX_ENTRY_FORMAT, slot4_data_offset, len(garbage), 0, len(b"corrupt.png"), 0, fn)
        raw[slot4_idx_offset : slot4_idx_offset + INDEX_ENTRY_SIZE] = entry

        path.write_bytes(bytes(raw))

        dest = tmp_path / "out"
        extracted = BlobExtractor(path).extract(dest)
        assert len(extracted) == 3
        assert {p.name for p in extracted} == {"valid_0.png", "valid_1.png", "valid_2.png"}


# ---------------------------------------------------------------------------
# Pattern resolution
# ---------------------------------------------------------------------------


class BlobPatternTest:
    def test_no_pattern_uses_literal_path(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "literal.blob"
        blob = BlobFsCapture(path, num_slots=4, max_image_bytes=256 * 1024)
        blob.close()
        assert path.exists()

# ---------------------------------------------------------------------------
# rotate()
# ---------------------------------------------------------------------------


class RotateTest:
    def test_blob_rotate_renames_file_to_event_id(self, tmp_path: pathlib.Path) -> None:
        original = tmp_path / "cap-XXX.blob"
        blob = BlobFsCapture(original, num_slots=4, max_image_bytes=512 * 1024)
        resolved = blob._blob_path  # e.g. cap-001.blob
        blob.capture(_make_image(), "frame0.png")
        old = blob.rotate("my-event-id")
        try:
            # The original file was renamed to my-event-id.blob
            assert (tmp_path / "my-event-id.blob").exists()
            assert old._blob_path == tmp_path / "my-event-id.blob"
            # The primary blob reopened at the original resolved path
            assert blob._blob_path == resolved
            assert resolved.exists()
        finally:
            old.close()
            blob.close()

    def test_blob_rotate_old_handle_retains_data(self, tmp_path: pathlib.Path) -> None:
        blob = BlobFsCapture(tmp_path / "cap-XXX.blob", num_slots=4, max_image_bytes=512 * 1024)
        blob.capture(_make_image(), "before_rotate.png")
        old = blob.rotate("event-abc")
        try:
            blob.capture(_make_image(), "after_rotate.png")
            blob.close()
            old.close()
            # Old handle (renamed to event-abc.blob): contains before_rotate.png
            dest = tmp_path / "from_old"
            extracted_old = BlobExtractor(tmp_path / "event-abc.blob").extract(dest)
            assert {p.name for p in extracted_old} == {"before_rotate.png"}
            # New blob (original path): contains after_rotate.png
            dest2 = tmp_path / "from_new"
            extracted_new = BlobExtractor(tmp_path / "cap-XXX.blob").extract(dest2)
            assert {p.name for p in extracted_new} == {"after_rotate.png"}
        finally:
            pass  # already closed above

    def test_blob_rotate_multiple_times(self, tmp_path: pathlib.Path) -> None:
        blob = BlobFsCapture(tmp_path / "cap-XX.blob", num_slots=4, max_image_bytes=512 * 1024)
        event_ids = ["evt-a", "evt-b", "evt-c"]
        handles = []
        for i, eid in enumerate(event_ids):
            blob.capture(_make_image(), f"frame{i}.png")
            handles.append(blob.rotate(eid))
        blob.close()
        for h in handles:
            h.close()
        # Three event blobs renamed + the primary still exists at cap-01.blob
        for eid in event_ids:
            assert (tmp_path / f"{eid}.blob").exists()
        assert (tmp_path / "cap-XX.blob").exists()


# ---------------------------------------------------------------------------
# CompositeBlobCapture
# ---------------------------------------------------------------------------


class CompositeBlobCaptureTest:
    def _make_composite(
        self,
        tmp_path: pathlib.Path,
        follow_up: int = 2,
        handler: BlobCompletionHandler | None = None,
    ) -> CompositeBlobCapture:
        if handler is None:
            handler = NoOpBlobCompletionHandler()
        return CompositeBlobCapture(
            tmp_path / "cap-XXX.blob",
            num_slots=8,
            max_image_bytes=512 * 1024,
            follow_up_count=follow_up,
            completion_handler=handler,
        )

    def test_capture_writes_to_primary(self, tmp_path: pathlib.Path) -> None:
        comp = self._make_composite(tmp_path)
        comp.capture(_make_image(), "frame.png")
        comp.close()
        blobs = list(tmp_path.glob("*.blob"))
        assert len(blobs) == 1
        extracted = BlobExtractor(blobs[0]).extract(tmp_path / "out")
        assert len(extracted) == 1

    def test_rotate_renames_file_to_event_id(self, tmp_path: pathlib.Path) -> None:
        comp = self._make_composite(tmp_path)
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("fire-uuid")
        comp.close()
        assert (tmp_path / "fire-uuid.blob").exists()

    def test_follow_up_writes_reach_rotated_blob(self, tmp_path: pathlib.Path) -> None:
        follow_up = 3
        comp = self._make_composite(tmp_path, follow_up=follow_up)
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("ev1")
        for i in range(follow_up):
            comp.capture(_make_image(), f"followup{i}.png")
        comp.close()
        # Event blob should contain: trigger frame + follow_up frames
        extracted = BlobExtractor(tmp_path / "ev1.blob").extract(tmp_path / "out")
        assert len(extracted) == 1 + follow_up

    def test_completion_handler_called_with_event_id_and_blob(self, tmp_path: pathlib.Path) -> None:
        handler = MagicMock(spec=BlobCompletionHandler)
        follow_up = 2
        comp = self._make_composite(tmp_path, follow_up=follow_up, handler=handler)
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("ev-done")
        for _ in range(follow_up):
            comp.capture(_make_image(), "extra.png")
        comp.close()
        handler.assert_called_once()
        call_event_id, call_blob = handler.call_args[0]
        assert call_event_id == "ev-done"
        assert isinstance(call_blob, BlobFsCapture)

    def test_no_completion_on_regular_close(self, tmp_path: pathlib.Path) -> None:
        handler = MagicMock(spec=BlobCompletionHandler)
        comp = self._make_composite(tmp_path, follow_up=5, handler=handler)
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("ev-early")
        # close without exhausting follow-up quota
        comp.capture(_make_image(), "one.png")
        comp.close()
        handler.assert_not_called()

    def test_trigger_frame_flag_in_primary_pre_rotation(self, tmp_path: pathlib.Path) -> None:
        comp = self._make_composite(tmp_path, follow_up=2)
        comp.capture(_make_image(), "trigger.png", flags=TRIGGER_FRAME_FLAG)
        primary_path = comp._blob._blob_path
        comp.rotate("ev-flag")
        comp.close()
        # The trigger frame is in the event blob (the file that was the primary before rotation)
        data = (tmp_path / "ev-flag.blob").read_bytes()
        _img_off, _img_sz, flags, *_ = struct.unpack_from(INDEX_ENTRY_FORMAT, data, SUPERBLOCK_SIZE)
        assert flags == TRIGGER_FRAME_FLAG
        # New primary blob has no entries yet (fresh)
        primary_data = primary_path.read_bytes()
        _magic2, _ver2, _slots2, wh2, *_ = struct.unpack_from(SUPERBLOCK_FORMAT, primary_data, 0)
        assert wh2 == 0

    def test_follow_up_entries_have_zero_flags(self, tmp_path: pathlib.Path) -> None:
        follow_up = 2
        comp = self._make_composite(tmp_path, follow_up=follow_up)
        comp.capture(_make_image(), "trigger.png", flags=TRIGGER_FRAME_FLAG)
        comp.rotate("ev-flags")
        for i in range(follow_up):
            comp.capture(_make_image(), f"followup{i}.png")
        comp.close()
        data = (tmp_path / "ev-flags.blob").read_bytes()
        # Check all follow-up entries have flags == 0
        for slot in range(1, 1 + follow_up):
            offset = SUPERBLOCK_SIZE + slot * INDEX_ENTRY_SIZE
            _img_off, _img_sz, flags, *_ = struct.unpack_from(INDEX_ENTRY_FORMAT, data, offset)
            assert flags == 0

    def test_update_last_flags_sets_flag_on_primary_blob(self, tmp_path: pathlib.Path) -> None:
        comp = self._make_composite(tmp_path)
        comp.capture(_make_image(), "frame.png", flags=0)
        comp.update_last_flags(TRIGGER_FRAME_FLAG)
        primary_path = comp._blob._blob_path
        comp.close()
        data = primary_path.read_bytes()
        _img_off, _img_sz, flags, *_ = struct.unpack_from(INDEX_ENTRY_FORMAT, data, SUPERBLOCK_SIZE)
        assert flags == TRIGGER_FRAME_FLAG

    def test_active_blob_path_returns_primary_path(self, tmp_path: pathlib.Path) -> None:
        comp = self._make_composite(tmp_path)
        assert comp.active_blob_path() == comp._blob._blob_path
        comp.close()

    def test_active_blob_path_updates_after_rotate(self, tmp_path: pathlib.Path) -> None:
        comp = self._make_composite(tmp_path)
        original_primary = comp._blob._blob_path
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("ev-rotate")
        # After rotation the primary blob has been reopened at the original path
        assert comp.active_blob_path() == original_primary
        comp.close()

    def test_rotate_immediate_calls_handler_right_away(self, tmp_path: pathlib.Path) -> None:
        handler = MagicMock(spec=BlobCompletionHandler)
        comp = self._make_composite(tmp_path, follow_up=3, handler=handler)
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("ev-imm", immediate=True)
        # Handler must be called immediately — no follow-up frames needed
        handler.assert_called_once()
        call_event_id, call_blob = handler.call_args[0]
        assert call_event_id == "ev-imm"
        assert isinstance(call_blob, BlobFsCapture)
        comp.close()

    def test_rotate_immediate_does_not_add_to_active(self, tmp_path: pathlib.Path) -> None:
        comp = self._make_composite(tmp_path, follow_up=3)
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("ev-imm2", immediate=True)
        assert len(comp._active) == 0
        comp.close()

    def test_rotate_immediate_does_not_deliver_follow_up_frames(self, tmp_path: pathlib.Path) -> None:
        handler = MagicMock(spec=BlobCompletionHandler)
        comp = self._make_composite(tmp_path, follow_up=3, handler=handler)
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("ev-imm3", immediate=True)
        # Additional frames go only to the new primary — handler stays called exactly once
        for i in range(3):
            comp.capture(_make_image(), f"extra{i}.png")
        comp.close()
        handler.assert_called_once()


# ---------------------------------------------------------------------------
# add_args / make_capture_backend
# ---------------------------------------------------------------------------


class CaptureFactoryTest:
    def test_add_args_registers_defaults(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args([])
        assert args.blob_capture_file == "capture.blob"
        assert args.blob_num_slots == 60
        assert args.blob_max_image_bytes == 4 * 1024 * 1024
        assert args.blob_png_compression == 1

    def test_add_args_allows_overrides(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args(
            ["--blob-capture-file", "my.blob", "--blob-num-slots", "10", "--blob-max-image-bytes", "1024"]
        )
        assert args.blob_capture_file == "my.blob"
        assert args.blob_num_slots == 10
        assert args.blob_max_image_bytes == 1024

    def test_make_capture_backend_returns_composite(self, tmp_path: pathlib.Path) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args([])
        backend = make_capture_backend(args, tmp_path, follow_up_count=5)
        assert isinstance(backend, CompositeBlobCapture)
        backend.close()

    def test_make_capture_backend_uses_completion_handler(self, tmp_path: pathlib.Path) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args([])
        handler = MagicMock(spec=BlobCompletionHandler)
        backend = make_capture_backend(args, tmp_path, follow_up_count=1, completion_handler=handler)
        assert isinstance(backend, CompositeBlobCapture)
        backend.capture(_make_image(), "t.png")
        backend.rotate("ev")
        backend.capture(_make_image(), "f.png")
        backend.close()
        handler.assert_called_once()


# ---------------------------------------------------------------------------
# blob-extract CLI — trigger frame annotation
# ---------------------------------------------------------------------------


def _run_extract_cli(blob_path: pathlib.Path, dest: pathlib.Path, info: bool = False) -> str:
    """Run the extract CLI and return captured stdout."""
    stdout = io.StringIO()
    argv = ["blob-extract", str(blob_path), str(dest)]
    if info:
        argv.append("--info")
    with patch("sys.argv", argv), patch("sys.stdout", stdout), patch("sys.stderr", io.StringIO()):
        blob_extract_main()
    return stdout.getvalue()


class BlobExtractCliTest:
    def _write_blob(self, tmp_path: pathlib.Path) -> pathlib.Path:
        path = tmp_path / "test.blob"
        blob = BlobFsCapture(path, num_slots=4, max_image_bytes=512 * 1024)
        blob.capture(_make_image(), "before.png", flags=0)
        blob.capture(_make_image(), "trigger.png", flags=TRIGGER_FRAME_FLAG)
        blob.capture(_make_image(), "after.png", flags=0)
        blob.close()
        return path

    def test_trigger_frame_line_is_annotated(self, tmp_path: pathlib.Path) -> None:
        blob_path = self._write_blob(tmp_path)
        output = _run_extract_cli(blob_path, tmp_path / "out")
        lines = {line.strip() for line in output.splitlines() if line.strip()}
        assert any("trigger.png" in line and "[TRIGGER_FRAME_FLAG]" in line for line in lines)

    def test_non_trigger_lines_have_no_annotation(self, tmp_path: pathlib.Path) -> None:
        blob_path = self._write_blob(tmp_path)
        output = _run_extract_cli(blob_path, tmp_path / "out")
        for line in output.splitlines():
            if "before.png" in line or "after.png" in line:
                assert "[TRIGGER_FRAME_FLAG]" not in line

    def test_no_trigger_frame_no_annotation(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "plain.blob"
        blob = BlobFsCapture(path, num_slots=4, max_image_bytes=512 * 1024)
        blob.capture(_make_image(), "frame0.png", flags=0)
        blob.capture(_make_image(), "frame1.png", flags=0)
        blob.close()
        output = _run_extract_cli(path, tmp_path / "out")
        assert "[TRIGGER_FRAME_FLAG]" not in output

    def test_blob_info(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "info.blob"
        blob = BlobFsCapture(path, num_slots=4, max_image_bytes=512 * 1024)
        blob.capture(_make_image(), "frame0.png", flags=0)
        blob.capture(_make_image(), "frame1.png", flags=TRIGGER_FRAME_FLAG)
        blob.close()
        output = _run_extract_cli(path, tmp_path / "out", info=True)

        assert f"blob:             {path}" in output
        assert f"magic:            {MAGIC:#010x}" in output
        assert "version:          1" in output
        assert "num_slots:        4  (write_head=2)" in output
        assert "max_image_bytes:  512.0 KiB (524288 bytes)" in output
        assert "occupied_slots:   2/4" in output
        assert "trigger_frame: frame1.png" in output
        assert not (tmp_path / "out").exists()
