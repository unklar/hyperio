from __future__ import annotations

import json
import logging
import pathlib
import re
import struct
import xml.etree.ElementTree as ET
from itertools import permutations
from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio
import spectral.io.envi as envi
import tifffile
from PIL import Image

from ._utils import (
    ensure_wavelengths_match_channels,
    normalize_to_hwc,
    parse_envi_wavelengths,
    to_float32_cube,
    try_parse_wavelengths_from_xml_like_text,
)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadResult:
    cube: np.ndarray
    wavelengths: np.ndarray
    reference_spectrum: np.ndarray | None = None
    reference_multiplier: float = 1.0
    reference_eps: float = 1e-8


@dataclass(frozen=True)
class Jp2Metadata:
    wavelengths: np.ndarray | None
    shape: tuple[int, int, int] | None
    reference_spectrum: np.ndarray | None
    reference_multiplier: float
    reference_eps: float


def _sidecar_json_path(image_path: pathlib.Path) -> pathlib.Path:
    """Return the JSON sidecar path for a given image path.

    For regular files (e.g. ``cube.hdr``) the sidecar is ``cube.json``.
    For directories (PNG folder) the sidecar is ``<dir>/metadata.json``.
    """
    if image_path.is_dir() or not image_path.suffix:
        return image_path / "metadata.json"
    return image_path.with_suffix(".json")


def _write_sidecar_json(
    image_path: pathlib.Path,
    wavelengths: np.ndarray,
    reference_spectrum: np.ndarray | None = None,
    reference_multiplier: float = 1.0,
    reference_eps: float = 1e-8,
) -> pathlib.Path:
    """Write a JSON sidecar file with full HSI metadata."""
    meta: dict[str, Any] = {
        "wavelengths": wavelengths.astype(np.float64).tolist(),
    }
    if reference_spectrum is not None:
        meta["reference_spectrum"] = reference_spectrum.astype(np.float64).tolist()
    if reference_multiplier != 1.0:
        meta["reference_multiplier"] = reference_multiplier
    if reference_eps != 1e-8:
        meta["reference_eps"] = reference_eps

    json_path = _sidecar_json_path(image_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(meta, indent=2))
    return json_path


def _read_sidecar_json(image_path: pathlib.Path) -> dict[str, Any] | None:
    """Read a JSON sidecar file, returning None if it does not exist."""
    json_path = _sidecar_json_path(image_path)
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text())


def _extract_png_index(stem: str) -> int | None:
    """Extract numeric index from PNG stem.

    Supported patterns:
    - 00000
    - some_name_00000
    """
    m_numeric = re.fullmatch(r"(\d+)", stem)
    if m_numeric is not None:
        return int(m_numeric.group(1))

    m_suffix = re.fullmatch(r".+_(\d+)", stem)
    if m_suffix is not None:
        return int(m_suffix.group(1))

    return None


def _resolve_wavelengths(
    cube: np.ndarray,
    *,
    explicit_wavelengths: np.ndarray | list[float] | None,
    min_wavelength: float | None,
    max_wavelength: float | None,
    extracted_wavelengths: np.ndarray | None,
    format_name: str,
) -> np.ndarray:
    channels = cube.shape[2]

    has_bounds = min_wavelength is not None or max_wavelength is not None
    if has_bounds and (min_wavelength is None or max_wavelength is None):
        raise ValueError("Both min_wavelength and max_wavelength must be provided together")

    if explicit_wavelengths is not None and has_bounds:
        raise ValueError("Provide either wavelengths or min/max wavelength, not both")

    if explicit_wavelengths is not None:
        return ensure_wavelengths_match_channels(np.asarray(explicit_wavelengths), channels)

    if has_bounds:
        return np.linspace(float(min_wavelength), float(max_wavelength), channels, dtype=np.float32)

    if extracted_wavelengths is None:
        raise ValueError(
            f"{format_name} does not provide valid wavelength metadata. "
            "Pass wavelengths explicitly or use min_wavelength/max_wavelength."
        )

    return ensure_wavelengths_match_channels(extracted_wavelengths, channels)


def _align_wavelength_count(wavelengths: np.ndarray, channels: int) -> np.ndarray:
    """Align parsed wavelength arrays to channel count when metadata contains extra values."""
    wl = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
    if wl.size == channels:
        return wl
    if wl.size < channels:
        return wl

    if channels > 1:
        increasing = np.diff(wl) > 0
        window = channels - 1
        if increasing.size >= window:
            conv = np.convolve(increasing.astype(np.int32), np.ones(window, dtype=np.int32), mode="valid")
            starts = np.where(conv == window)[0]
            for start in starts:
                seg = wl[start : start + channels]
                if seg.size == channels and 250.0 <= float(seg.min()) <= 3000.0 and 250.0 <= float(seg.max()) <= 3000.0:
                    return seg

    _logger.warning(
        "_align_wavelength_count: extracted %d wavelength values but cube has %d channels; "
        "truncating to first %d values. Pass wavelengths explicitly or use metadata_json=True "
        "to avoid this.",
        wl.size, channels, channels,
    )
    return wl[:channels]


def _canonicalize_reference_for_unit_cube(reference: np.ndarray, cube: np.ndarray) -> np.ndarray:
    """Normalize reference spectrum scale to unit domain when cube is already in [0,1]."""
    ref = np.asarray(reference, dtype=np.float32).copy()
    if ref.size == 0:
        return ref

    ref_min = float(np.min(ref))
    ref_max = float(np.max(ref))
    if ref_min < 0.0 or ref_max <= 1.0:
        return ref

    cube_min = float(np.min(cube))
    cube_max = float(np.max(cube))
    if cube_min >= 0.0 and cube_max <= 1.0 and ref_max <= 255.0:
        ref /= 255.0
    return ref


def _normalize_cube_with_reference(
    cube: np.ndarray,
    reference_spectrum: np.ndarray,
    reference_multiplier: float = 1.0,
    reference_eps: float = 1e-8,
) -> np.ndarray:
    """Normalize cube by reference spectrum band-wise."""
    ref = np.asarray(reference_spectrum, dtype=np.float32).reshape(-1)
    if ref.size != cube.shape[2]:
        raise ValueError(
            f"reference_spectrum size ({ref.size}) does not match channels ({cube.shape[2]})"
        )

    eps = float(reference_eps) if float(reference_eps) > 0.0 else 1e-8
    mult = float(reference_multiplier) if float(reference_multiplier) > 0.0 else 1.0
    denom = ref * np.float32(mult)
    safe = np.where(np.abs(denom) >= eps, denom, np.where(denom < 0.0, -eps, eps)).astype(np.float32)
    return (cube / safe[None, None, :]).astype(np.float32)


def _parse_payload_dict(text: str) -> dict[str, Any] | None:
    """Parse JSON object or XML key/value metadata payload into a dictionary."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        root = ET.fromstring(text)
    except Exception:
        return None

    parsed: dict[str, Any] = {}
    for node in list(root):
        key = node.tag
        raw = node.text or ""
        try:
            parsed[key] = json.loads(raw)
        except Exception:
            parsed[key] = raw
    return parsed if parsed else None


def _parse_shape_from_text(text: str) -> tuple[int, int, int] | None:
    """Parse [H, W, C]-like shape metadata from JSON/XML text."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            raw_shape = obj.get("shape")
            if isinstance(raw_shape, list) and len(raw_shape) == 3:
                vals = tuple(int(v) for v in raw_shape)
                if all(v > 0 for v in vals):
                    return vals
    except Exception:
        pass

    m = re.search(r"shape[^\d]{0,40}(\d+)[^\d]+(\d+)[^\d]+(\d+)", text, flags=re.IGNORECASE)
    if m is not None:
        vals = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if all(v > 0 for v in vals):
            return vals
    return None


def _reorder_hwc_to_target_shape(cube: np.ndarray, target_shape: tuple[int, int, int] | None) -> np.ndarray:
    """Permute HWC axes to match target shape if possible."""
    if target_shape is None:
        return cube

    cur = cube.shape
    if cur == target_shape:
        return cube

    for perm in permutations((0, 1, 2), 3):
        cand_shape = (cur[perm[0]], cur[perm[1]], cur[perm[2]])
        if cand_shape == target_shape:
            return np.transpose(cube, axes=perm)

    return cube


def _read_u32_be(data: bytes, off: int) -> int:
    return struct.unpack(">I", data[off : off + 4])[0]


def _read_u64_be(data: bytes, off: int) -> int:
    return struct.unpack(">Q", data[off : off + 8])[0]


def extract_jp2_metadata(path: str | pathlib.Path) -> Jp2Metadata:
    """Extract wavelengths, shape and optional normalization metadata from JP2 XML/UUID payloads."""
    data = pathlib.Path(path).read_bytes()
    off = 0
    best_wavelengths: np.ndarray | None = None
    target_shape: tuple[int, int, int] | None = None
    reference_spectrum: np.ndarray | None = None
    reference_multiplier: float = 1.0
    reference_eps: float = 1e-8

    while off + 8 <= len(data):
        lbox = _read_u32_be(data, off)
        tbox = _read_u32_be(data, off + 4)
        header_size = 8

        if lbox == 1:
            if off + 16 > len(data):
                break
            box_size = _read_u64_be(data, off + 8)
            header_size = 16
        elif lbox == 0:
            box_size = len(data) - off
        else:
            box_size = lbox

        if box_size < header_size or off + box_size > len(data):
            break

        payload = data[off + header_size : off + box_size]

        if tbox in (0x786D6C20, 0x75756964):
            try:
                as_text = payload.decode("utf-8", errors="ignore")
                payload_dict = _parse_payload_dict(as_text)
                if payload_dict is not None:
                    if reference_spectrum is None and "reference_spectrum" in payload_dict:
                        try:
                            reference_spectrum = np.asarray(payload_dict["reference_spectrum"], dtype=np.float32)
                        except Exception:
                            pass
                    if "reference_multiplier" in payload_dict:
                        try:
                            reference_multiplier = float(payload_dict["reference_multiplier"])
                        except Exception:
                            pass
                    if "reference_eps" in payload_dict:
                        try:
                            reference_eps = float(payload_dict["reference_eps"])
                        except Exception:
                            pass

                from_text = try_parse_wavelengths_from_xml_like_text(as_text)
                if from_text is not None:
                    best_wavelengths = from_text if best_wavelengths is None else best_wavelengths

                parsed_shape = _parse_shape_from_text(as_text)
                if parsed_shape is not None and target_shape is None:
                    target_shape = parsed_shape

                obj = json.loads(as_text)
                if isinstance(obj, dict) and "wavelength" in obj:
                    best_wavelengths = np.asarray(obj["wavelength"], dtype=np.float32)
                if isinstance(obj, dict) and "shape" in obj and target_shape is None:
                    raw_shape = obj.get("shape")
                    if isinstance(raw_shape, list) and len(raw_shape) == 3:
                        target_shape = (int(raw_shape[0]), int(raw_shape[1]), int(raw_shape[2]))
            except Exception:
                pass

            try:
                root = ET.fromstring(payload.decode("utf-8", errors="ignore"))
                xml_text = ET.tostring(root, encoding="unicode")
                from_xml = try_parse_wavelengths_from_xml_like_text(xml_text)
                if from_xml is not None:
                    best_wavelengths = from_xml if best_wavelengths is None else best_wavelengths
                parsed_shape = _parse_shape_from_text(xml_text)
                if parsed_shape is not None and target_shape is None:
                    target_shape = parsed_shape
            except Exception:
                pass

        off += box_size

    return Jp2Metadata(
        wavelengths=best_wavelengths,
        shape=target_shape,
        reference_spectrum=reference_spectrum,
        reference_multiplier=reference_multiplier,
        reference_eps=reference_eps,
    )


def read_envi(
    path: str | pathlib.Path,
    wavelengths: np.ndarray | list[float] | None = None,
    min_wavelength: float | None = None,
    max_wavelength: float | None = None,
    metadata_json: bool = False,
) -> ReadResult:
    p = pathlib.Path(path)
    sidecar_wavelengths: np.ndarray | None = None
    sidecar_ref: np.ndarray | None = None
    sidecar_mult: float = 1.0
    sidecar_eps: float = 1e-8
    if metadata_json:
        sc = _read_sidecar_json(p)
        if sc is not None:
            if "wavelengths" in sc:
                sidecar_wavelengths = np.asarray(sc["wavelengths"], dtype=np.float32)
            if "reference_spectrum" in sc:
                sidecar_ref = np.asarray(sc["reference_spectrum"], dtype=np.float32)
            if "reference_multiplier" in sc:
                sidecar_mult = float(sc["reference_multiplier"])
            if "reference_eps" in sc:
                sidecar_eps = float(sc["reference_eps"])

    img = envi.open(str(path))
    cube = np.asarray(img.load())
    cube = normalize_to_hwc(cube)
    cube = to_float32_cube(cube)

    try:
        extracted = parse_envi_wavelengths(img.metadata.get("wavelength"))
    except ValueError:
        extracted = None
    if sidecar_wavelengths is not None:
        extracted = sidecar_wavelengths
    wl = _resolve_wavelengths(
        cube,
        explicit_wavelengths=wavelengths,
        min_wavelength=min_wavelength,
        max_wavelength=max_wavelength,
        extracted_wavelengths=extracted,
        format_name="ENVI",
    )
    return ReadResult(
        cube=cube,
        wavelengths=wl,
        reference_spectrum=sidecar_ref,
        reference_multiplier=sidecar_mult,
        reference_eps=sidecar_eps,
    )


def read_tiff(
    path: str | pathlib.Path,
    wavelengths: np.ndarray | list[float] | None = None,
    min_wavelength: float | None = None,
    max_wavelength: float | None = None,
    metadata_json: bool = False,
) -> ReadResult:
    p = pathlib.Path(path)
    sidecar_wavelengths: np.ndarray | None = None
    sidecar_ref: np.ndarray | None = None
    sidecar_mult: float = 1.0
    sidecar_eps: float = 1e-8
    if metadata_json:
        sc = _read_sidecar_json(p)
        if sc is not None:
            if "wavelengths" in sc:
                sidecar_wavelengths = np.asarray(sc["wavelengths"], dtype=np.float32)
            if "reference_spectrum" in sc:
                sidecar_ref = np.asarray(sc["reference_spectrum"], dtype=np.float32)
            if "reference_multiplier" in sc:
                sidecar_mult = float(sc["reference_multiplier"])
            if "reference_eps" in sc:
                sidecar_eps = float(sc["reference_eps"])

    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()

        extracted: np.ndarray | None = None
        meta_candidates: list[Any] = []
        if tif.imagej_metadata is not None:
            meta_candidates.append(tif.imagej_metadata)
        if tif.shaped_metadata is not None:
            meta_candidates.append(tif.shaped_metadata)
        for page in tif.pages:
            if page.description:
                meta_candidates.append(page.description)

        for cand in meta_candidates:
            if isinstance(cand, dict):
                for key in ("wavelength", "wavelengths", "wavelength_nm"):
                    if key in cand:
                        try:
                            extracted = np.asarray(cand[key], dtype=np.float32)
                            break
                        except Exception:
                            continue
                if extracted is not None:
                    break
            elif isinstance(cand, (list, tuple)):
                for item in cand:
                    if isinstance(item, dict):
                        for key in ("wavelength", "wavelengths", "wavelength_nm"):
                            if key in item:
                                try:
                                    extracted = np.asarray(item[key], dtype=np.float32)
                                    break
                                except Exception:
                                    continue
                        if extracted is not None:
                            break
                if extracted is not None:
                    break
            else:
                cand_str = str(cand)
                parsed_dict = _parse_payload_dict(cand_str)
                if parsed_dict is not None:
                    for key in ("wavelength", "wavelengths", "wavelength_nm"):
                        if key in parsed_dict:
                            try:
                                extracted = np.asarray(parsed_dict[key], dtype=np.float32)
                                break
                            except Exception:
                                continue
                if extracted is None:
                    parsed = try_parse_wavelengths_from_xml_like_text(cand_str)
                    if parsed is not None:
                        extracted = parsed
                if extracted is not None:
                    break

    cube = normalize_to_hwc(np.asarray(arr))
    cube = to_float32_cube(cube)

    if sidecar_wavelengths is not None:
        extracted = sidecar_wavelengths
    wl = _resolve_wavelengths(
        cube,
        explicit_wavelengths=wavelengths,
        min_wavelength=min_wavelength,
        max_wavelength=max_wavelength,
        extracted_wavelengths=extracted,
        format_name="TIFF",
    )
    return ReadResult(
        cube=cube,
        wavelengths=wl,
        reference_spectrum=sidecar_ref,
        reference_multiplier=sidecar_mult,
        reference_eps=sidecar_eps,
    )


def read_jp2(
    path: str | pathlib.Path,
    wavelengths: np.ndarray | list[float] | None = None,
    min_wavelength: float | None = None,
    max_wavelength: float | None = None,
    normalize: bool = True,
    metadata_json: bool = False,
) -> ReadResult:
    p = pathlib.Path(path)
    sidecar_wavelengths: np.ndarray | None = None
    sidecar_ref: np.ndarray | None = None
    sidecar_mult: float = 1.0
    sidecar_eps: float = 1e-8
    if metadata_json:
        sc = _read_sidecar_json(p)
        if sc is not None:
            if "wavelengths" in sc:
                sidecar_wavelengths = np.asarray(sc["wavelengths"], dtype=np.float32)
            if "reference_spectrum" in sc:
                sidecar_ref = np.asarray(sc["reference_spectrum"], dtype=np.float32)
            if "reference_multiplier" in sc:
                sidecar_mult = float(sc["reference_multiplier"])
            if "reference_eps" in sc:
                sidecar_eps = float(sc["reference_eps"])

    with rasterio.open(path) as src:
        arr = src.read()

    if arr.ndim != 3:
        raise ValueError(f"JP2 reader expected 3D raster data, got shape {arr.shape}")

    cube = np.moveaxis(arr, 0, -1)

    md = extract_jp2_metadata(path)
    cube = _reorder_hwc_to_target_shape(cube, md.shape)
    cube = to_float32_cube(cube)

    extracted = md.wavelengths
    if sidecar_wavelengths is not None:
        extracted = sidecar_wavelengths
    if extracted is None:
        try:
            with rasterio.open(path) as src:
                tags = src.tags(1)
                wl_str = tags.get("wavelength")
                if wl_str is not None:
                    extracted = np.asarray(json.loads(wl_str), dtype=np.float32)
        except Exception:
            pass
    if extracted is not None:
        extracted = _align_wavelength_count(extracted, int(cube.shape[2]))

    if normalize and md.reference_spectrum is not None and sidecar_ref is None:
        ref = _align_wavelength_count(md.reference_spectrum, int(cube.shape[2]))
        if ref.size != cube.shape[2]:
            raise ValueError(
                f"reference_spectrum size ({ref.size}) does not match channels ({cube.shape[2]})"
            )
        ref = _canonicalize_reference_for_unit_cube(ref, cube)
        cube = _normalize_cube_with_reference(
            cube,
            reference_spectrum=ref,
            reference_multiplier=md.reference_multiplier,
            reference_eps=md.reference_eps,
        )
    elif normalize and sidecar_ref is not None:
        ref = _align_wavelength_count(sidecar_ref, int(cube.shape[2]))
        ref = _canonicalize_reference_for_unit_cube(ref, cube)
        cube = _normalize_cube_with_reference(
            cube,
            reference_spectrum=ref,
            reference_multiplier=sidecar_mult,
            reference_eps=sidecar_eps,
        )

    wl = _resolve_wavelengths(
        cube,
        explicit_wavelengths=wavelengths,
        min_wavelength=min_wavelength,
        max_wavelength=max_wavelength,
        extracted_wavelengths=extracted,
        format_name="JP2",
    )
    return ReadResult(
        cube=cube,
        wavelengths=wl,
        reference_spectrum=sidecar_ref if sidecar_ref is not None else md.reference_spectrum,
        reference_multiplier=sidecar_mult if sidecar_ref is not None else md.reference_multiplier,
        reference_eps=sidecar_eps if sidecar_ref is not None else md.reference_eps,
    )


def read_hsd(
    path: str | pathlib.Path,
    wavelengths: np.ndarray | list[float] | None = None,
    min_wavelength: float | None = None,
    max_wavelength: float | None = None,
    metadata_json: bool = False,
) -> ReadResult:
    """Read HSICityV2 HSD files and reconstruct the hyperspectral cube.

    Also supports a simplified raw-cube variant (d=0) written by
    ``write_hsd``, where the full float32 cube is stored verbatim
    after the header.
    """
    p = pathlib.Path(path)
    sidecar_wavelengths: np.ndarray | None = None
    sidecar_ref: np.ndarray | None = None
    sidecar_mult: float = 1.0
    sidecar_eps: float = 1e-8
    if metadata_json:
        sc = _read_sidecar_json(p)
        if sc is not None:
            if "wavelengths" in sc:
                sidecar_wavelengths = np.asarray(sc["wavelengths"], dtype=np.float32)
            if "reference_spectrum" in sc:
                sidecar_ref = np.asarray(sc["reference_spectrum"], dtype=np.float32)
            if "reference_multiplier" in sc:
                sidecar_mult = float(sc["reference_multiplier"])
            if "reference_eps" in sc:
                sidecar_eps = float(sc["reference_eps"])
    header = np.fromfile(path, dtype=np.int32, count=7)
    if header.size != 7:
        raise ValueError("Invalid HSD file: could not read 7 int32 header values")

    height = int(header[0])
    width = int(header[1])
    sr = int(header[2])
    d = int(header[3])
    startw = float(header[4])
    endw = float(header[6])

    if height <= 0 or width <= 0 or sr <= 0:
        raise ValueError("Invalid HSD header values: height, width and SR must be > 0")

    if d == 0:
        total_floats_raw = 1 + sr + height * width * sr
        float_data = np.fromfile(path, dtype=np.float32, count=total_floats_raw, offset=7 * 4)
        idx = 0
        _stepw = float_data[idx]
        idx += 1
        _average = float_data[idx : idx + sr]
        idx += sr
        cube = float_data[idx : idx + height * width * sr].reshape((height, width, sr))
        cube = to_float32_cube(cube)
    else:
        total_floats_with_step = 1 + sr + d * sr + height * width * d
        total_floats_no_step = sr + d * sr + height * width * d

        float_data = np.fromfile(path, dtype=np.float32, count=total_floats_with_step, offset=7 * 4)
        has_stepw = True

        if float_data.size == total_floats_no_step:
            has_stepw = False
        elif float_data.size != total_floats_with_step:
            raise ValueError(
                "Invalid HSD file size: "
                f"expected {total_floats_with_step} (with stepw) or {total_floats_no_step} (without stepw) "
                f"float32 values after header, got {float_data.size}"
            )

        idx = 0
        if has_stepw:
            _stepw = float_data[idx]
            idx += 1
        else:
            _stepw = np.float32(np.nan)

        average = float_data[idx : idx + sr]
        idx += sr

        coeff = float_data[idx : idx + d * sr].reshape((d, sr))
        idx += d * sr

        scoredata = float_data[idx : idx + height * width * d].reshape((height * width, d))

        temp = np.dot(scoredata, coeff)
        cube = (temp + average).reshape((height, width, sr))
        cube = to_float32_cube(cube)

    extracted = np.linspace(startw, endw, sr, dtype=np.float32)
    if sidecar_wavelengths is not None:
        extracted = sidecar_wavelengths
    wl = _resolve_wavelengths(
        cube,
        explicit_wavelengths=wavelengths,
        min_wavelength=min_wavelength,
        max_wavelength=max_wavelength,
        extracted_wavelengths=extracted,
        format_name="HSD",
    )

    return ReadResult(
        cube=cube,
        wavelengths=wl,
        reference_spectrum=sidecar_ref,
        reference_multiplier=sidecar_mult,
        reference_eps=sidecar_eps,
    )


def read_line_scan_png_folder(
    folder: str | pathlib.Path,
    wavelengths: np.ndarray | list[float] | None = None,
    min_wavelength: float | None = None,
    max_wavelength: float | None = None,
    line_cam: bool = True,
    metadata_json: bool = False,
) -> ReadResult:
    """Read a folder of PNG files into a hyperspectral cube.

    Supported file names:
    - Numeric: 00000.png, 00001.png, ...
    - Suffix index: some_name_00000.png, some_name_00001.png, ...

    If line_cam=True, each image is expected to represent one line (height, spectrum).
    If line_cam=False, each image is expected to represent one spectral channel (height, width).
    """
    p = pathlib.Path(folder)
    if not p.is_dir():
        raise ValueError(f"Path is not a directory: {p}")

    sidecar_wavelengths: np.ndarray | None = None
    sidecar_ref: np.ndarray | None = None
    sidecar_mult: float = 1.0
    sidecar_eps: float = 1e-8
    if metadata_json:
        sc = _read_sidecar_json(p)
        if sc is not None:
            if "wavelengths" in sc:
                sidecar_wavelengths = np.asarray(sc["wavelengths"], dtype=np.float32)
            if "reference_spectrum" in sc:
                sidecar_ref = np.asarray(sc["reference_spectrum"], dtype=np.float32)
            if "reference_multiplier" in sc:
                sidecar_mult = float(sc["reference_multiplier"])
            if "reference_eps" in sc:
                sidecar_eps = float(sc["reference_eps"])

    numbered_pngs: list[tuple[int, pathlib.Path]] = []
    for file_path in p.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() != ".png":
            continue
        stem = file_path.stem
        idx = _extract_png_index(stem)
        if idx is None:
            continue
        numbered_pngs.append((idx, file_path))

    if not numbered_pngs:
        raise ValueError(
            "No indexed PNG files found in directory. Expected names like 00000.png or some_name_00000.png."
        )

    has_explicit_wavelengths = wavelengths is not None
    has_bounds = min_wavelength is not None and max_wavelength is not None
    if min_wavelength is not None and max_wavelength is None:
        raise ValueError("Both min_wavelength and max_wavelength must be provided together")
    if min_wavelength is None and max_wavelength is not None:
        raise ValueError("Both min_wavelength and max_wavelength must be provided together")
    if has_explicit_wavelengths and has_bounds:
        raise ValueError("Provide either wavelengths or min/max wavelength, not both")

    wl_len = int(np.asarray(wavelengths).shape[0]) if has_explicit_wavelengths else None

    numbered_pngs.sort(key=lambda item: item[0])

    if line_cam:
        n_lines = len(numbered_pngs)
        expected_shape: tuple[int, int] | None = None
        first_arr: np.ndarray | None = None
        for idx, (_, file_path) in enumerate(numbered_pngs):
            with Image.open(file_path) as img:
                arr = np.asarray(img)
            if arr.ndim != 2:
                raise ValueError(
                    f"Line-scan PNG must be 2D (height, spectrum), got shape {arr.shape} in {file_path.name}"
                )
            if wl_len is not None and arr.shape[1] == wl_len:
                line = arr
            elif wl_len is not None and arr.shape[0] == wl_len:
                line = arr.T
            elif wl_len is None:
                line = arr
            else:
                raise ValueError(
                    f"Could not determine spectral axis for {file_path.name}: shape {arr.shape} "
                    f"does not match wavelength count {wl_len} in any axis"
                )

            if expected_shape is None:
                expected_shape = (int(line.shape[0]), int(line.shape[1]))
                cube = np.empty((expected_shape[0], n_lines, expected_shape[1]), dtype=line.dtype)
            elif line.shape != expected_shape:
                raise ValueError(
                    f"All line-scan PNG files must have identical normalized shape. "
                    f"Expected {expected_shape}, got {line.shape} in {file_path.name}"
                )

            cube[:, idx, :] = line
    else:
        n_channels = len(numbered_pngs)
        expected_shape = None
        for idx, (_, file_path) in enumerate(numbered_pngs):
            with Image.open(file_path) as img:
                arr = np.asarray(img)
            if arr.ndim != 2:
                raise ValueError(
                    f"Channel PNG must be 2D (height, width), got shape {arr.shape} in {file_path.name}"
                )

            if expected_shape is None:
                expected_shape = (int(arr.shape[0]), int(arr.shape[1]))
                cube = np.empty((expected_shape[0], expected_shape[1], n_channels), dtype=arr.dtype)
            elif arr.shape != expected_shape:
                raise ValueError(
                    f"All channel PNG files must have identical shape. "
                    f"Expected {expected_shape}, got {arr.shape} in {file_path.name}"
                )

            cube[:, :, idx] = arr

    cube = to_float32_cube(cube)

    extracted_wl: np.ndarray | None = None
    if sidecar_wavelengths is not None:
        extracted_wl = sidecar_wavelengths
    wl = _resolve_wavelengths(
        cube,
        explicit_wavelengths=wavelengths,
        min_wavelength=min_wavelength,
        max_wavelength=max_wavelength,
        extracted_wavelengths=extracted_wl,
        format_name="Line-scan PNG folder",
    )
    return ReadResult(
        cube=cube,
        wavelengths=wl,
        reference_spectrum=sidecar_ref,
        reference_multiplier=sidecar_mult,
        reference_eps=sidecar_eps,
    )


def read_auto(
    path: str | pathlib.Path,
    wavelengths: np.ndarray | list[float] | None = None,
    min_wavelength: float | None = None,
    max_wavelength: float | None = None,
    line_cam: bool = True,
    normalize: bool = True,
    metadata_json: bool = False,
) -> ReadResult:
    p = pathlib.Path(path)
    if p.is_dir():
        return read_line_scan_png_folder(
            p,
            wavelengths=wavelengths,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
            line_cam=line_cam,
            metadata_json=metadata_json,
        )

    ext = p.suffix.lower()

    if ext == ".hdr":
        return read_envi(
            p,
            wavelengths=wavelengths,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
            metadata_json=metadata_json,
        )
    if ext in {".tif", ".tiff"}:
        return read_tiff(
            p,
            wavelengths=wavelengths,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
            metadata_json=metadata_json,
        )
    if ext == ".jp2":
        return read_jp2(
            p,
            wavelengths=wavelengths,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
            normalize=normalize,
            metadata_json=metadata_json,
        )
    if ext == ".hsd":
        return read_hsd(
            p,
            wavelengths=wavelengths,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
            metadata_json=metadata_json,
        )

    raise ValueError(f"Unsupported file extension: {ext}")


def write_envi(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    path: str | pathlib.Path,
    metadata_json: bool = False,
    reference_spectrum: np.ndarray | None = None,
    reference_multiplier: float = 1.0,
    reference_eps: float = 1e-8,
) -> pathlib.Path:
    """Write an HSI cube to ENVI format (raw binary + .hdr header).

    The .hdr file stores wavelength metadata. The raw file uses the same
    stem with no extension (ENVI convention).  Returns the path to the
    .hdr file.
    """
    p = pathlib.Path(path)
    hdr_path = p if p.suffix.lower() == ".hdr" else p.with_suffix(".hdr")
    hdr_path.parent.mkdir(parents=True, exist_ok=True)

    h, w, c = cube.shape
    meta = {
        "lines": h,
        "samples": w,
        "bands": c,
        "header_offset": 0,
        "file_type": "ENVI Standard",
        "data_type": 4,
        "interleave": "bsq",
        "byte_order": 0,
        "wavelength": wavelengths.tolist(),
    }

    envi.save_image(str(hdr_path), cube, metadata=meta, interleave="bsq")

    if metadata_json:
        _write_sidecar_json(
            hdr_path,
            wavelengths,
            reference_spectrum=reference_spectrum,
            reference_multiplier=reference_multiplier,
            reference_eps=reference_eps,
        )

    return hdr_path


def write_tiff(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    path: str | pathlib.Path,
    metadata_json: bool = False,
    reference_spectrum: np.ndarray | None = None,
    reference_multiplier: float = 1.0,
    reference_eps: float = 1e-8,
) -> pathlib.Path:
    """Write an HSI cube to TIFF with wavelength metadata embedded.

    Wavelengths are stored in the ImageJ metadata tag so they survive
    a round-trip through ``read_tiff``.  Returns the path written.
    """
    p = pathlib.Path(path)
    if p.suffix.lower() not in {".tif", ".tiff"}:
        p = p.with_suffix(".tiff")
    p.parent.mkdir(parents=True, exist_ok=True)

    metadata = {"wavelengths": wavelengths.astype(np.float64).tolist()}
    tifffile.imwrite(str(p), cube, metadata=metadata)

    if metadata_json:
        _write_sidecar_json(
            p,
            wavelengths,
            reference_spectrum=reference_spectrum,
            reference_multiplier=reference_multiplier,
            reference_eps=reference_eps,
        )

    return p


def write_jp2(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    path: str | pathlib.Path,
    metadata_json: bool = False,
    reference_spectrum: np.ndarray | None = None,
    reference_multiplier: float = 1.0,
    reference_eps: float = 1e-8,
) -> pathlib.Path:
    """Write an HSI cube to JPEG 2000.

    Wavelength metadata is stored via rasterio band tags (key
    ``"wavelength"`` on band 1).  Because JP2 is a lossy format,
    round-trip fidelity is not exact.  Returns the path written.
    """
    p = pathlib.Path(path)
    if p.suffix.lower() != ".jp2":
        p = p.with_suffix(".jp2")
    p.parent.mkdir(parents=True, exist_ok=True)

    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    h, w, c = cube.shape
    if np.any(cube > 1.0) or np.any(cube < 0.0):
        _logger.warning(
            "write_jp2: cube contains values outside [0, 1]; "
            "they will be clipped during uint16 conversion."
        )
    cube_uint16 = np.clip(cube * 65535.0, 0, 65535).astype(np.uint16)
    transform = from_bounds(0, 0, w, h, w, h)
    with rasterio.open(
        str(p),
        "w",
        driver="JP2OpenJPEG",
        height=h,
        width=w,
        count=c,
        dtype="uint16",
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        for band_idx in range(c):
            dst.write(cube_uint16[:, :, band_idx], band_idx + 1)
        wl_json = json.dumps(wavelengths.tolist())
        dst.update_tags(1, wavelength=wl_json)

    if metadata_json:
        _write_sidecar_json(
            p,
            wavelengths,
            reference_spectrum=reference_spectrum,
            reference_multiplier=reference_multiplier,
            reference_eps=reference_eps,
        )

    return p


def write_hsd(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    path: str | pathlib.Path,
    metadata_json: bool = False,
    reference_spectrum: np.ndarray | None = None,
    reference_multiplier: float = 1.0,
    reference_eps: float = 1e-8,
) -> pathlib.Path:
    """Write an HSI cube to HSICityV2-compatible HSD format.

    The cube is written as raw float32 data following a 7-int32 header.
    Wavelength endpoints (startw, endw) are stored in the header.
    Since no PCA compression is applied, the ``d`` header field is set
    to 0 as a sentinel and the full ``height * width * sr`` float32
    values are written after the header.  Returns the path written.

    Note: this simplified variant is not compatible with the
    PCA-compressed HSICityV2 format.  It can be round-tripped through
    ``read_hsd`` within this library, but other HSICityV2 readers will
    not understand it.
    """
    p = pathlib.Path(path)
    if p.suffix.lower() != ".hsd":
        p = p.with_suffix(".hsd")
    p.parent.mkdir(parents=True, exist_ok=True)

    h, w, c = cube.shape
    startw = float(wavelengths[0])
    endw = float(wavelengths[-1])

    header = np.array([h, w, c, 0, round(startw), 0, round(endw)], dtype=np.int32)

    stepw = np.array([(endw - startw) / max(c - 1, 1)], dtype=np.float32)
    average = np.zeros(c, dtype=np.float32)

    with p.open("wb") as f:
        header.tofile(f)
        stepw.tofile(f)
        average.tofile(f)
        cube.reshape(-1).astype(np.float32).tofile(f)

    if metadata_json:
        _write_sidecar_json(
            p,
            wavelengths,
            reference_spectrum=reference_spectrum,
            reference_multiplier=reference_multiplier,
            reference_eps=reference_eps,
        )

    return p


def write_png_folder(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    folder: str | pathlib.Path,
    line_cam: bool = True,
    metadata_json: bool = False,
    reference_spectrum: np.ndarray | None = None,
    reference_multiplier: float = 1.0,
    reference_eps: float = 1e-8,
) -> pathlib.Path:
    """Write an HSI cube as a folder of indexed PNG files.

    Each PNG represents one scan line (``line_cam=True``) or one
    spectral channel (``line_cam=False``).  File names are zero-padded
    integers: ``00000.png``, ``00001.png``, ...

    Values outside ``[0, 1]`` are clipped before conversion to uint16.
    Because PNG does not store wavelength metadata, callers must
    persist wavelengths separately.  Returns the folder path.
    """
    p = pathlib.Path(folder)
    p.mkdir(parents=True, exist_ok=True)

    if np.any(cube > 1.0) or np.any(cube < 0.0):
        _logger.warning(
            "write_png_folder: cube contains values outside [0, 1]; "
            "they will be clipped during uint16 conversion."
        )
    if not metadata_json:
        _logger.warning(
            "write_png_folder: PNG files do not store wavelength metadata. "
            "Use metadata_json=True to save a JSON sidecar, or pass "
            "wavelengths= explicitly when reading."
        )

    if line_cam:
        for i in range(cube.shape[1]):
            line = cube[:, i, :]
            line_uint16 = np.clip(line * 65535.0, 0, 65535).astype(np.uint16)
            img = Image.fromarray(line_uint16)
            img.save(p / f"{i:05d}.png")
    else:
        for i in range(cube.shape[2]):
            channel = cube[:, :, i]
            channel_uint16 = np.clip(channel * 65535.0, 0, 65535).astype(np.uint16)
            img = Image.fromarray(channel_uint16)
            img.save(p / f"{i:05d}.png")

    if metadata_json:
        _write_sidecar_json(
            p,
            wavelengths,
            reference_spectrum=reference_spectrum,
            reference_multiplier=reference_multiplier,
            reference_eps=reference_eps,
        )

    return p


def write_auto(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    path: str | pathlib.Path,
    metadata_json: bool = False,
    reference_spectrum: np.ndarray | None = None,
    reference_multiplier: float = 1.0,
    reference_eps: float = 1e-8,
    line_cam: bool = True,
) -> pathlib.Path:
    """Dispatch to the appropriate writer based on file extension or path type.

    Supported extensions: ``.hdr`` (ENVI), ``.tif``/``.tiff``, ``.jp2``,
    ``.hsd``.  If *path* is a directory, ``write_png_folder`` is used.
    """
    p = pathlib.Path(path)

    if p.is_dir() or (not p.suffix):
        return write_png_folder(
            cube, wavelengths, p,
            line_cam=line_cam,
            metadata_json=metadata_json,
            reference_spectrum=reference_spectrum,
            reference_multiplier=reference_multiplier,
            reference_eps=reference_eps,
        )

    ext = p.suffix.lower()

    kw = dict(
        metadata_json=metadata_json,
        reference_spectrum=reference_spectrum,
        reference_multiplier=reference_multiplier,
        reference_eps=reference_eps,
    )

    if ext == ".hdr":
        return write_envi(cube, wavelengths, p, **kw)
    if ext in {".tif", ".tiff"}:
        return write_tiff(cube, wavelengths, p, **kw)
    if ext == ".jp2":
        return write_jp2(cube, wavelengths, p, **kw)
    if ext == ".hsd":
        return write_hsd(cube, wavelengths, p, **kw)

    raise ValueError(
        f"Unsupported file extension for writing: {ext}. "
        "Supported: .hdr (ENVI), .tif/.tiff, .jp2, .hsd, or a directory (PNG folder)"
    )
