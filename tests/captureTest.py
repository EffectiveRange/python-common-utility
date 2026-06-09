import argparse
import io
import pathlib
import struct
import tarfile

import numpy as np
import pytest

from unittest.mock import MagicMock, patch

import cv2

from common_utility.capture import (
    MAGIC,
    SUPERBLOCK_FORMAT,
    SUPERBLOCK_SIZE,
    INDEX_ENTRY_FORMAT,
    INDEX_ENTRY_SIZE,
    TRIGGER_FRAME_FLAG,
    RAW_IMAGE_FLAG,
    RAW_IMAGE_HEADER_FORMAT,
    RAW_IMAGE_HEADER_SIZE,
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

    def test_open_on_existing_path_creates_fresh_blob(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "cap.blob"
        blob = BlobFsCapture(path, num_slots=4, max_image_bytes=256 * 1024)
        blob.close()
        raw = bytearray(path.read_bytes())
        struct.pack_into("<I", raw, 0, 0xDEADBEEF)
        path.write_bytes(bytes(raw))
        reopened = BlobFsCapture(path, num_slots=4, max_image_bytes=256 * 1024)
        reopened.close()
        data = path.read_bytes()
        magic = struct.unpack_from("<I", data, 0)[0]
        assert magic == MAGIC

    def test_write_and_ring_wrap(self, tmp_path: pathlib.Path) -> None:
        num_slots = 3
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=num_slots, max_image_bytes=512 * 1024)
        images = [_make_image() for _ in range(num_slots + 1)]
        filenames = [f"frame_{i}.png" for i in range(num_slots + 1)]
        for img, fn in zip(images, filenames):
            blob.capture(img, fn)
        blob.close()

        data = (tmp_path / "cap.blob").read_bytes()
        _, _, _, write_head, _, _ = struct.unpack_from(SUPERBLOCK_FORMAT, data, 0)
        assert write_head == 1

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

    def test_update_last_flags_sets_flags_field(self, tmp_path: pathlib.Path) -> None:
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=4, max_image_bytes=512 * 1024)
        blob.capture(_make_image(), "frame.png", flags=0)
        blob.update_last_flags(TRIGGER_FRAME_FLAG)
        blob.close()
        data = (tmp_path / "cap.blob").read_bytes()
        _img_off, _img_sz, flags, _fn_len, _res, _fn_raw = struct.unpack_from(INDEX_ENTRY_FORMAT, data, SUPERBLOCK_SIZE)
        assert flags == TRIGGER_FRAME_FLAG

    def test_raw_capture_sets_raw_image_flag(self, tmp_path: pathlib.Path) -> None:
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=4, max_image_bytes=512 * 1024, raw_capture=True)
        blob.capture(_make_image(), "frame.png")
        blob.close()
        data = (tmp_path / "cap.blob").read_bytes()
        _img_off, _img_sz, flags, _fn_len, _res, _fn_raw = struct.unpack_from(INDEX_ENTRY_FORMAT, data, SUPERBLOCK_SIZE)
        assert flags & RAW_IMAGE_FLAG

    def test_raw_capture_stores_header_and_pixels(self, tmp_path: pathlib.Path) -> None:
        image = _make_image()
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=4, max_image_bytes=512 * 1024, raw_capture=True)
        blob.capture(image, "frame.png")
        blob.close()
        data = (tmp_path / "cap.blob").read_bytes()
        img_off, img_sz, *_ = struct.unpack_from(INDEX_ENTRY_FORMAT, data, SUPERBLOCK_SIZE)
        raw = data[img_off : img_off + img_sz]
        assert len(raw) == RAW_IMAGE_HEADER_SIZE + image.size
        h, w, c = struct.unpack_from(RAW_IMAGE_HEADER_FORMAT, raw, 0)
        assert (h, w, c) == (image.shape[0], image.shape[1], image.shape[2])

    def test_update_last_flags_preserves_raw_image_flag(self, tmp_path: pathlib.Path) -> None:
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=4, max_image_bytes=512 * 1024, raw_capture=True)
        blob.capture(_make_image(), "frame.png")
        blob.update_last_flags(TRIGGER_FRAME_FLAG)
        blob.close()
        data = (tmp_path / "cap.blob").read_bytes()
        _img_off, _img_sz, flags, *_ = struct.unpack_from(INDEX_ENTRY_FORMAT, data, SUPERBLOCK_SIZE)
        assert flags == (RAW_IMAGE_FLAG | TRIGGER_FRAME_FLAG)

    def test_open_on_size_mismatch_creates_fresh_blob(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "cap.blob"
        blob = BlobFsCapture(path, num_slots=4, max_image_bytes=256 * 1024)
        blob.close()
        with open(path, "ab") as f:
            f.write(b"\x00")
        reopened = BlobFsCapture(path, num_slots=4, max_image_bytes=256 * 1024)
        reopened.close()
        expected = SUPERBLOCK_SIZE + 4 * INDEX_ENTRY_SIZE + 4 * 256 * 1024
        assert path.stat().st_size == expected

    def test_get_last_filepath_returns_none(self, tmp_path: pathlib.Path) -> None:
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=4, max_image_bytes=256 * 1024)
        assert blob.get_last_filepath() is None
        blob.close()

    def test_set_capture_ts_is_a_no_op(self, tmp_path: pathlib.Path) -> None:
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=4, max_image_bytes=256 * 1024)
        blob.set_capture_ts(MagicMock())
        blob.close()

    def test_active_blob_path_returns_blob_path(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "cap.blob"
        blob = BlobFsCapture(path, num_slots=4, max_image_bytes=256 * 1024)
        assert blob.active_blob_path() == path
        blob.close()

    def test_open_always_starts_with_zero_write_head(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "cap.blob"
        blob = BlobFsCapture(path, num_slots=4, max_image_bytes=256 * 1024)
        blob.capture(_make_image(), "frame.png")
        blob.close()
        reopened = BlobFsCapture(path, num_slots=4, max_image_bytes=256 * 1024)
        assert reopened._write_head == 0
        reopened.close()

    def test_double_close_is_safe(self, tmp_path: pathlib.Path) -> None:
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=4, max_image_bytes=256 * 1024)
        blob.close()
        blob.close()

    def test_update_last_flags_targets_most_recent_slot(self, tmp_path: pathlib.Path) -> None:
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=4, max_image_bytes=512 * 1024)
        blob.capture(_make_image(), "first.png", flags=0)
        blob.capture(_make_image(), "second.png", flags=0)
        blob.update_last_flags(TRIGGER_FRAME_FLAG)
        blob.close()
        data = (tmp_path / "cap.blob").read_bytes()
        _a, _b, flags0, *_ = struct.unpack_from(INDEX_ENTRY_FORMAT, data, SUPERBLOCK_SIZE)
        assert flags0 == 0
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

    def test_truncated_data_section_extracts_intact_images(self, tmp_path: pathlib.Path) -> None:
        """Blob truncated partway into the third slot's data: first two images are still extracted."""
        num_slots = 3
        max_bytes = 512 * 1024
        path = tmp_path / "truncated.blob"
        blob = BlobFsCapture(path, num_slots=num_slots, max_image_bytes=max_bytes)
        for name in ["a.png", "b.png", "c.png"]:
            blob.capture(_make_image(), name)
        blob.close()

        data_section_start = SUPERBLOCK_SIZE + num_slots * INDEX_ENTRY_SIZE
        path.write_bytes(path.read_bytes()[: data_section_start + 2 * max_bytes + 50])

        extracted = BlobExtractor(path).extract(tmp_path / "out")
        assert len(extracted) == 2
        assert {p.name for p in extracted} == {"a.png", "b.png"}

    def test_double_close_is_safe(self, tmp_path: pathlib.Path) -> None:
        path = self._write_blob(tmp_path, [(_make_image(), "frame.png")])
        e = BlobExtractor(path)
        e.close()
        e.close()

    def test_file_too_small_raises(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "tiny.blob"
        path.write_bytes(b"\x00" * (SUPERBLOCK_SIZE - 1))
        with pytest.raises(ValueError, match="too small"):
            BlobExtractor(path)

    def test_context_manager_and_truncated_index_breaks_early(self, tmp_path: pathlib.Path) -> None:
        """Index truncated mid-second-entry: extract() and get_filenames_with_flags() both break early."""
        path = tmp_path / "cap.blob"
        blob = BlobFsCapture(path, num_slots=4, max_image_bytes=512 * 1024)
        blob.capture(_make_image(), "frame.png")
        blob.close()
        truncate_at = SUPERBLOCK_SIZE + INDEX_ENTRY_SIZE + INDEX_ENTRY_SIZE // 2
        path.write_bytes(path.read_bytes()[:truncate_at])
        with BlobExtractor(path) as e:
            extracted = e.extract(tmp_path / "out")
            trigger_names = e.get_filenames_with_flags(TRIGGER_FRAME_FLAG)
        assert extracted == []
        assert trigger_names == set()
        assert e._mm is None  # context manager closed it

    def test_zero_filename_len_uses_slot_index_fallback(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "cap.blob"
        blob = BlobFsCapture(path, num_slots=4, max_image_bytes=512 * 1024)
        blob.capture(_make_image(), "real.png")
        blob.close()
        raw = bytearray(path.read_bytes())
        struct.pack_into("<I", raw, SUPERBLOCK_SIZE + 24, 0)
        path.write_bytes(bytes(raw))
        extracted = BlobExtractor(path).extract(tmp_path / "out")
        assert len(extracted) == 1
        assert extracted[0].name == "slot_0000.png"

    def test_latin1_filename_decoded_in_extract_and_frame_flags(self, tmp_path: pathlib.Path) -> None:
        """Filename bytes invalid in UTF-8 but valid in latin-1 are decoded correctly in both paths."""
        path = tmp_path / "cap.blob"
        blob = BlobFsCapture(path, num_slots=4, max_image_bytes=512 * 1024)
        blob.capture(_make_image(), "placeholder.png")
        blob.close()
        latin1_name = b"caf\xe9.png"
        raw = bytearray(path.read_bytes())
        struct.pack_into("<I", raw, SUPERBLOCK_SIZE + 24, len(latin1_name))
        raw[SUPERBLOCK_SIZE + 32 : SUPERBLOCK_SIZE + 32 + len(latin1_name)] = latin1_name
        path.write_bytes(bytes(raw))
        with BlobExtractor(path) as e:
            extracted = e.extract(tmp_path / "out")
            e.get_filenames_with_flags(TRIGGER_FRAME_FLAG)
        assert len(extracted) == 1
        assert extracted[0].name == "café.png"

    def test_corruption_resilience(self, tmp_path: pathlib.Path) -> None:
        """Simulates two interrupted writes; BlobExtractor must extract the 3 valid images."""
        path = tmp_path / "cap.blob"
        blob = BlobFsCapture(path, num_slots=5, max_image_bytes=512 * 1024)
        for i in range(3):
            blob.capture(_make_image(), f"valid_{i}.png")
        blob.close()

        raw = bytearray(path.read_bytes())

        struct.pack_into("<I", raw, 12, 4)

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

    def test_sibling_blobs_combined_on_extract(self, tmp_path: pathlib.Path) -> None:
        """A primary blob and a -post sibling are both extracted and combined transparently."""
        primary = tmp_path / "event-abc.blob"
        post = tmp_path / "event-abc-post.blob"

        b1 = BlobFsCapture(primary, num_slots=4, max_image_bytes=512 * 1024)
        b1.capture(_make_image(), "pre.png")
        b1.close()

        b2 = BlobFsCapture(post, num_slots=4, max_image_bytes=512 * 1024)
        b2.capture(_make_image(), "post.png")
        b2.close()

        extracted = BlobExtractor(primary).extract(tmp_path / "out")
        assert {p.name for p in extracted} == {"pre.png", "post.png"}

    def test_no_siblings_when_none_exist(self, tmp_path: pathlib.Path) -> None:
        """When no sibling blobs exist, only the primary blob is extracted."""
        path = tmp_path / "solo.blob"
        b = BlobFsCapture(path, num_slots=4, max_image_bytes=512 * 1024)
        b.capture(_make_image(), "only.png")
        b.close()

        extracted = BlobExtractor(path).extract(tmp_path / "out")
        assert {p.name for p in extracted} == {"only.png"}

    def test_raw_capture_extractor_produces_png(self, tmp_path: pathlib.Path) -> None:
        names = ["a.png", "b.png"]
        path = tmp_path / "cap.blob"
        blob = BlobFsCapture(path, num_slots=4, max_image_bytes=512 * 1024, raw_capture=True)
        for name in names:
            blob.capture(_make_image(), name)
        blob.close()
        dest = tmp_path / "out"
        extracted = BlobExtractor(path).extract(dest)
        assert len(extracted) == 2
        assert {p.name for p in extracted} == set(names)
        for p in extracted:
            arr = cv2.imdecode(np.frombuffer(p.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            assert arr is not None

    def test_raw_capture_round_trip_pixel_equality(self, tmp_path: pathlib.Path) -> None:
        image = _make_image()
        path = tmp_path / "cap.blob"
        blob = BlobFsCapture(path, num_slots=4, max_image_bytes=512 * 1024, raw_capture=True)
        blob.capture(image, "frame.png")
        blob.close()
        extracted = BlobExtractor(path).extract(tmp_path / "out")
        assert len(extracted) == 1
        decoded = cv2.imdecode(np.frombuffer(extracted[0].read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        # cv2 stores BGR; compare channel-by-channel (lossless PNG round-trip must be identical)
        assert np.array_equal(image, decoded)

    def test_raw_and_png_blobs_extracted_transparently(self, tmp_path: pathlib.Path) -> None:
        """Primary blob uses PNG, post sibling uses raw; both extract to valid PNGs."""
        primary = tmp_path / "event.blob"
        post = tmp_path / "event-post.blob"
        b1 = BlobFsCapture(primary, num_slots=4, max_image_bytes=512 * 1024, raw_capture=False)
        b1.capture(_make_image(), "png.png")
        b1.close()
        b2 = BlobFsCapture(post, num_slots=4, max_image_bytes=512 * 1024, raw_capture=True)
        b2.capture(_make_image(), "raw.png")
        b2.close()
        extracted = BlobExtractor(primary).extract(tmp_path / "out")
        assert {p.name for p in extracted} == {"png.png", "raw.png"}
        for p in extracted:
            arr = cv2.imdecode(np.frombuffer(p.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            assert arr is not None


# ---------------------------------------------------------------------------
# BlobExtractor — tar/tar.gz archive support
# ---------------------------------------------------------------------------


class BlobExtractorTarTest:
    def _write_blob(self, tmp_path: pathlib.Path, images_and_names: list[tuple]) -> pathlib.Path:
        path = tmp_path / "cap.blob"
        blob = BlobFsCapture(path, num_slots=5, max_image_bytes=512 * 1024)
        for img, fn in images_and_names:
            blob.capture(img, fn)
        blob.close()
        return path

    def test_extract_from_tar(self, tmp_path: pathlib.Path) -> None:
        names = ["a.png", "b.png", "c.png"]
        blob_path = self._write_blob(tmp_path, [(_make_image(), n) for n in names])
        tar_path = tmp_path / "cap.tar"
        with tarfile.open(tar_path, "w") as tf:
            tf.add(blob_path, arcname="cap.blob")
        dest = tmp_path / "out"
        extracted = BlobExtractor(tar_path).extract(dest)
        assert len(extracted) == 3
        assert {p.name for p in extracted} == set(names)

    def test_extract_from_tar_gz(self, tmp_path: pathlib.Path) -> None:
        names = ["x.png", "y.png"]
        blob_path = self._write_blob(tmp_path, [(_make_image(), n) for n in names])
        tar_path = tmp_path / "cap.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(blob_path, arcname="cap.blob")
        dest = tmp_path / "out"
        extracted = BlobExtractor(tar_path).extract(dest)
        assert len(extracted) == 2
        assert {p.name for p in extracted} == set(names)

    def test_tar_prefers_blob_extension_members(self, tmp_path: pathlib.Path) -> None:
        """Archive with a non-blob file and a blob file: only the blob is extracted."""
        blob_path = self._write_blob(tmp_path, [(_make_image(), "frame.png")])
        dummy = tmp_path / "readme.txt"
        dummy.write_text("hello")
        tar_path = tmp_path / "multi.tar"
        with tarfile.open(tar_path, "w") as tf:
            tf.add(dummy, arcname="readme.txt")
            tf.add(blob_path, arcname="cap.blob")
        dest = tmp_path / "out"
        extracted = BlobExtractor(tar_path).extract(dest)
        assert len(extracted) == 1
        assert extracted[0].name == "frame.png"

    def test_multiple_blobs_in_tar_combined(self, tmp_path: pathlib.Path) -> None:
        """Tarball with two blob members: all images from both are combined."""
        pre_blob = tmp_path / "pre.blob"
        post_blob = tmp_path / "post.blob"

        b1 = BlobFsCapture(pre_blob, num_slots=4, max_image_bytes=512 * 1024)
        b1.capture(_make_image(), "pre.png")
        b1.close()

        b2 = BlobFsCapture(post_blob, num_slots=4, max_image_bytes=512 * 1024)
        b2.capture(_make_image(), "post.png")
        b2.close()

        tar_path = tmp_path / "event.tar"
        with tarfile.open(tar_path, "w") as tf:
            tf.add(pre_blob, arcname="event.blob")
            tf.add(post_blob, arcname="event-post.blob")

        extracted = BlobExtractor(tar_path).extract(tmp_path / "out")
        assert {p.name for p in extracted} == {"pre.png", "post.png"}

    def test_tar_tempdir_cleaned_up_on_close(self, tmp_path: pathlib.Path) -> None:
        blob_path = self._write_blob(tmp_path, [(_make_image(), "f.png")])
        tar_path = tmp_path / "cap.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(blob_path, arcname="cap.blob")
        e = BlobExtractor(tar_path)
        assert e._tempdir is not None
        tempdir_path = pathlib.Path(e._tempdir.name)
        assert tempdir_path.exists()
        e.close()
        assert not tempdir_path.exists()

    def test_tar_context_manager_cleans_up_tempdir(self, tmp_path: pathlib.Path) -> None:
        blob_path = self._write_blob(tmp_path, [(_make_image(), "g.png")])
        tar_path = tmp_path / "cap.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(blob_path, arcname="cap.blob")
        with BlobExtractor(tar_path) as e:
            assert e._tempdir is not None
            tempdir_path = pathlib.Path(e._tempdir.name)
        assert not tempdir_path.exists()

    def test_empty_tar_raises(self, tmp_path: pathlib.Path) -> None:
        tar_path = tmp_path / "empty.tar"
        with tarfile.open(tar_path, "w"):
            pass
        with pytest.raises(ValueError, match="No regular file"):
            BlobExtractor(tar_path)

    def test_non_tar_blob_unaffected(self, tmp_path: pathlib.Path) -> None:
        """A regular blob file is still opened directly without a tempdir."""
        blob_path = self._write_blob(tmp_path, [(_make_image(), "direct.png")])
        with BlobExtractor(blob_path) as e:
            assert e._tempdir is None
            extracted = e.extract(tmp_path / "out")
        assert len(extracted) == 1
        assert extracted[0].name == "direct.png"


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
    def test_blob_rotate_moves_to_capture_folder(self, tmp_path: pathlib.Path) -> None:
        cap_folder = tmp_path / "rotated"
        blob_path = tmp_path / "active.blob"
        blob = BlobFsCapture(blob_path, num_slots=4, max_image_bytes=512 * 1024, capture_folder=cap_folder)
        blob.capture(_make_image(), "frame0.png")
        blob.rotate("my-event-id")
        blob.close()
        assert (cap_folder / "my-event-id.blob").exists()
        assert blob_path.exists()  # fresh blob reopened at original path

    def test_blob_rotate_without_capture_folder_stays_local(self, tmp_path: pathlib.Path) -> None:
        blob_path = tmp_path / "cap.blob"
        blob = BlobFsCapture(blob_path, num_slots=4, max_image_bytes=512 * 1024)
        blob.capture(_make_image(), "frame0.png")
        blob.rotate("event-local")
        blob.close()
        assert (tmp_path / "event-local.blob").exists()
        assert blob_path.exists()

    def test_blob_rotate_data_is_in_capture_folder(self, tmp_path: pathlib.Path) -> None:
        cap_folder = tmp_path / "rotated"
        blob = BlobFsCapture(
            tmp_path / "active.blob", num_slots=4, max_image_bytes=512 * 1024, capture_folder=cap_folder
        )
        blob.capture(_make_image(), "before_rotate.png")
        blob.rotate("event-abc")
        blob.capture(_make_image(), "after_rotate.png")
        blob.close()

        extracted_old = BlobExtractor(cap_folder / "event-abc.blob").extract(tmp_path / "from_old")
        assert {p.name for p in extracted_old} == {"before_rotate.png"}

        extracted_new = BlobExtractor(tmp_path / "active.blob").extract(tmp_path / "from_new")
        assert {p.name for p in extracted_new} == {"after_rotate.png"}

    def test_blob_rotate_multiple_times(self, tmp_path: pathlib.Path) -> None:
        cap_folder = tmp_path / "rotated"
        blob = BlobFsCapture(
            tmp_path / "active.blob", num_slots=4, max_image_bytes=512 * 1024, capture_folder=cap_folder
        )
        event_ids = ["evt-a", "evt-b", "evt-c"]
        for i, eid in enumerate(event_ids):
            blob.capture(_make_image(), f"frame{i}.png")
            blob.rotate(eid)
        blob.close()
        for eid in event_ids:
            assert (cap_folder / f"{eid}.blob").exists()
        assert (tmp_path / "active.blob").exists()

    def test_blob_rotate_returns_self(self, tmp_path: pathlib.Path) -> None:
        blob = BlobFsCapture(tmp_path / "cap.blob", num_slots=4, max_image_bytes=512 * 1024)
        result = blob.rotate("ev")
        assert result is blob
        blob.close()


# ---------------------------------------------------------------------------
# CompositeBlobCapture
# ---------------------------------------------------------------------------


class CompositeBlobCaptureTest:
    def _make_composite(
        self,
        tmp_path: pathlib.Path,
        follow_up: int = 2,
        handler: BlobCompletionHandler | None = None,
        cap_folder: pathlib.Path | None = None,
    ) -> CompositeBlobCapture:
        if handler is None:
            handler = NoOpBlobCompletionHandler()
        if cap_folder is None:
            cap_folder = tmp_path / "rotated"
        return CompositeBlobCapture(
            tmp_path / "active.blob",
            cap_folder,
            num_slots=8,
            max_image_bytes=512 * 1024,
            follow_up_count=follow_up,
            completion_handler=handler,
        )

    def test_capture_writes_to_primary(self, tmp_path: pathlib.Path) -> None:
        comp = self._make_composite(tmp_path)
        comp.capture(_make_image(), "frame.png")
        comp.close()
        extracted = BlobExtractor(tmp_path / "active.blob").extract(tmp_path / "out")
        assert len(extracted) == 1

    def test_rotate_moves_primary_to_capture_folder(self, tmp_path: pathlib.Path) -> None:
        cap_folder = tmp_path / "rotated"
        comp = self._make_composite(tmp_path, cap_folder=cap_folder)
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("fire-uuid")
        comp.close()
        assert (cap_folder / "fire-uuid.blob").exists()

    def test_post_blob_receives_follow_up_frames(self, tmp_path: pathlib.Path) -> None:
        follow_up = 3
        cap_folder = tmp_path / "rotated"
        comp = self._make_composite(tmp_path, follow_up=follow_up, cap_folder=cap_folder)
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("ev1")
        for i in range(follow_up):
            comp.capture(_make_image(), f"followup{i}.png")
        comp.close()

        # Post blob alone (ev1-post.blob has no -* sibling): only follow_up frames
        post_extracted = BlobExtractor(cap_folder / "ev1-post.blob").extract(tmp_path / "post")
        assert len(post_extracted) == follow_up

        # BlobExtractor on the primary automatically includes the post sibling: trigger + follow-ups
        all_extracted = BlobExtractor(cap_folder / "ev1.blob").extract(tmp_path / "all")
        assert len(all_extracted) == 1 + follow_up

    def test_completion_handler_called_with_event_id_and_paths(self, tmp_path: pathlib.Path) -> None:
        handler = MagicMock(spec=BlobCompletionHandler)
        follow_up = 2
        cap_folder = tmp_path / "rotated"
        comp = self._make_composite(tmp_path, follow_up=follow_up, handler=handler, cap_folder=cap_folder)
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("ev-done")
        for _ in range(follow_up):
            comp.capture(_make_image(), "extra.png")
        comp.close()
        handler.assert_called_once()
        call_event_id, call_blob_path, call_additional = handler.call_args[0]
        assert call_event_id == "ev-done"
        assert call_blob_path == cap_folder / "ev-done.blob"
        assert call_additional == [cap_folder / "ev-done-post.blob"]

    def test_no_completion_on_regular_close(self, tmp_path: pathlib.Path) -> None:
        handler = MagicMock(spec=BlobCompletionHandler)
        comp = self._make_composite(tmp_path, follow_up=5, handler=handler)
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("ev-early")
        comp.capture(_make_image(), "one.png")
        comp.close()
        handler.assert_not_called()

    def test_trigger_frame_flag_in_pre_rotation_blob(self, tmp_path: pathlib.Path) -> None:
        cap_folder = tmp_path / "rotated"
        comp = self._make_composite(tmp_path, follow_up=2, cap_folder=cap_folder)
        comp.capture(_make_image(), "trigger.png", flags=TRIGGER_FRAME_FLAG)
        comp.rotate("ev-flag")
        comp.close()
        data = (cap_folder / "ev-flag.blob").read_bytes()
        _img_off, _img_sz, flags, *_ = struct.unpack_from(INDEX_ENTRY_FORMAT, data, SUPERBLOCK_SIZE)
        assert flags == TRIGGER_FRAME_FLAG

    def test_follow_up_entries_have_zero_flags(self, tmp_path: pathlib.Path) -> None:
        follow_up = 2
        cap_folder = tmp_path / "rotated"
        comp = self._make_composite(tmp_path, follow_up=follow_up, cap_folder=cap_folder)
        comp.capture(_make_image(), "trigger.png", flags=TRIGGER_FRAME_FLAG)
        comp.rotate("ev-flags")
        for i in range(follow_up):
            comp.capture(_make_image(), f"followup{i}.png")
        comp.close()
        data = (cap_folder / "ev-flags-post.blob").read_bytes()
        for slot in range(follow_up):
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

    def test_active_blob_path_unchanged_after_rotate(self, tmp_path: pathlib.Path) -> None:
        comp = self._make_composite(tmp_path)
        original_primary = comp._blob._blob_path
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("ev-rotate")
        assert comp.active_blob_path() == original_primary
        comp.close()

    def test_rotate_immediate_calls_handler_right_away(self, tmp_path: pathlib.Path) -> None:
        handler = MagicMock(spec=BlobCompletionHandler)
        cap_folder = tmp_path / "rotated"
        comp = self._make_composite(tmp_path, follow_up=3, handler=handler, cap_folder=cap_folder)
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("ev-imm", immediate=True)
        handler.assert_called_once()
        call_event_id, call_blob_path, call_additional = handler.call_args[0]
        assert call_event_id == "ev-imm"
        assert call_blob_path == cap_folder / "ev-imm.blob"
        assert call_additional == []
        comp.close()

    def test_rotate_immediate_no_post_blob_opened(self, tmp_path: pathlib.Path) -> None:
        cap_folder = tmp_path / "rotated"
        comp = self._make_composite(tmp_path, follow_up=3, cap_folder=cap_folder)
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("ev-imm2", immediate=True)
        assert comp._post_blob is None
        assert not (cap_folder / "ev-imm2-post.blob").exists()
        comp.close()

    def test_rotate_immediate_does_not_deliver_follow_up_frames(self, tmp_path: pathlib.Path) -> None:
        handler = MagicMock(spec=BlobCompletionHandler)
        comp = self._make_composite(tmp_path, follow_up=3, handler=handler)
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("ev-imm3", immediate=True)
        for i in range(3):
            comp.capture(_make_image(), f"extra{i}.png")
        comp.close()
        handler.assert_called_once()

    def test_rotate_when_post_blob_active_completes_existing(self, tmp_path: pathlib.Path) -> None:
        """A second rotation force-completes the existing post blob before starting a new one."""
        handler = MagicMock(spec=BlobCompletionHandler)
        cap_folder = tmp_path / "rotated"
        comp = self._make_composite(tmp_path, follow_up=5, handler=handler, cap_folder=cap_folder)
        comp.capture(_make_image(), "trigger1.png")
        comp.rotate("ev-first")
        # Only 1 follow-up before the second rotation
        comp.capture(_make_image(), "followup1.png")
        comp.rotate("ev-second")
        # ev-first's post blob should have been force-completed
        handler.assert_called_once()
        call_event_id, call_blob_path, call_additional = handler.call_args[0]
        assert call_event_id == "ev-first"
        comp.close()

    def test_set_capture_ts_delegates_to_primary(self, tmp_path: pathlib.Path) -> None:
        comp = self._make_composite(tmp_path)
        comp.set_capture_ts(MagicMock())
        comp.close()

    def test_get_last_filepath_delegates_to_primary(self, tmp_path: pathlib.Path) -> None:
        comp = self._make_composite(tmp_path)
        assert comp.get_last_filepath() is None
        comp.close()

    def test_composite_raw_capture_post_blob_also_raw(self, tmp_path: pathlib.Path) -> None:
        follow_up = 2
        cap_folder = tmp_path / "rotated"
        comp = CompositeBlobCapture(
            tmp_path / "active.blob",
            cap_folder,
            num_slots=8,
            max_image_bytes=512 * 1024,
            follow_up_count=follow_up,
            completion_handler=NoOpBlobCompletionHandler(),
            raw_capture=True,
        )
        comp.capture(_make_image(), "trigger.png")
        comp.rotate("ev-raw")
        for i in range(follow_up):
            comp.capture(_make_image(), f"followup{i}.png")
        comp.close()

        for blob_path in [cap_folder / "ev-raw.blob", cap_folder / "ev-raw-post.blob"]:
            data = blob_path.read_bytes()
            _, _, flags, *_ = struct.unpack_from(INDEX_ENTRY_FORMAT, data, SUPERBLOCK_SIZE)
            assert flags & RAW_IMAGE_FLAG

        all_extracted = BlobExtractor(cap_folder / "ev-raw.blob").extract(tmp_path / "out")
        assert len(all_extracted) == 1 + follow_up
        for p in all_extracted:
            arr = cv2.imdecode(np.frombuffer(p.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            assert arr is not None


# ---------------------------------------------------------------------------
# add_args / make_capture_backend
# ---------------------------------------------------------------------------


class CaptureFactoryTest:
    def test_add_args_registers_defaults(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args([])
        assert args.blob_capture_file == pathlib.Path("capture.blob")
        assert args.blob_capture_folder == pathlib.Path(".")
        assert args.blob_num_slots == 60
        assert args.blob_max_image_bytes == 4 * 1024 * 1024
        assert args.blob_png_compression == 1
        assert args.blob_raw_capture is False

    def test_add_args_raw_capture_flag(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args(["--blob-raw-capture"])
        assert args.blob_raw_capture is True

    def test_add_args_raw_capture_default_injectable(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser, blob_raw_capture=True)
        args = parser.parse_args([])
        assert args.blob_raw_capture is True

    def test_add_args_allows_overrides(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args(
            ["--blob-capture-file", "my.blob", "--blob-num-slots", "10", "--blob-max-image-bytes", "1024"]
        )
        assert args.blob_capture_file == pathlib.Path("my.blob")
        assert args.blob_num_slots == 10
        assert args.blob_max_image_bytes == 1024

    def test_make_capture_backend_returns_composite(self, tmp_path: pathlib.Path) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        cap_folder = tmp_path / "rotated"
        args = parser.parse_args(
            [
                f"--blob-capture-file={tmp_path / 'active.blob'}",
                f"--blob-capture-folder={cap_folder}",
            ]
        )
        backend = make_capture_backend(args, follow_up_count=5)
        assert isinstance(backend, CompositeBlobCapture)
        backend.close()

    def test_make_capture_backend_uses_completion_handler(self, tmp_path: pathlib.Path) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        cap_folder = tmp_path / "rotated"
        cap_folder.mkdir()
        args = parser.parse_args(
            [
                f"--blob-capture-file={tmp_path / 'active.blob'}",
                f"--blob-capture-folder={cap_folder}",
            ]
        )
        handler = MagicMock(spec=BlobCompletionHandler)
        backend = make_capture_backend(args, follow_up_count=1, completion_handler=handler)
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
    argv = ["blob-extract", "-d", str(dest)]
    if info:
        argv.append("--info")
    argv.append(str(blob_path))
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

    def test_missing_blob_exits_with_error(self, tmp_path: pathlib.Path) -> None:
        stderr = io.StringIO()
        argv = ["blob-extract", "-d", str(tmp_path / "out"), str(tmp_path / "nonexistent.blob")]
        with patch("sys.argv", argv), patch("sys.stderr", stderr):
            with pytest.raises(SystemExit) as exc:
                blob_extract_main()
        assert exc.value.code == 1
        assert "not found" in stderr.getvalue()

    def test_invalid_blob_exits_with_error(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "bad.blob"
        path.write_bytes(b"\xff" * SUPERBLOCK_SIZE)
        stderr = io.StringIO()
        argv = ["blob-extract", "-d", str(tmp_path / "out"), str(path)]
        with patch("sys.argv", argv), patch("sys.stderr", stderr):
            with pytest.raises(SystemExit) as exc:
                blob_extract_main()
        assert exc.value.code == 1
        assert "error" in stderr.getvalue()

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
