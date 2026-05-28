"""
Preprocess MU-Glioma-Post to TaDiff-Net npy format.

Input:
- dataset/MU-Glioma-Post
- dataset/MU-Glioma-Post_ClinicalData-FINAL032025.xlsx

Output per patient (matches data/README.md conventions):
- {patient_id}_image.npy: (M*T, H, W, D), M=4 modalities [T1, T1c, FLAIR, T2]
- {patient_id}_label.npy: (T, H, W, D)
- {patient_id}_days.npy: (T,), cumulative days rebased to start at 0
- {patient_id}_treatment.npy: (T,), binary treatment code
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np


# Label conventions aligned with existing SAILOR setup
BG = 0
EDEMA = 1
NECROTIC = 2
ENHANCING = 3


CLINICAL_SHEET_NAME = "MU Glioma Post"

# Timepoint day columns in the provided clinical xlsx.
TIMEPOINT_DAY_COLUMNS = {
    1: "Number of Days from Diagnosis to 1st MRI (Timepoint_1)",
    2: "Number of Days from Diagnosis to 2nd MRI (Timepoint_2)",
    3: "Number of Days from Diagnosis to 3rd MRI (Timepoint_3)",
    4: "Number of Days from Diagnosis to 4th MRI (Timepoint_4)",
    5: "Number of Days from Diagnosis to 5th MRI (Timepoint_5)",
    6: "Number of Days from Diagnosis to 6th MRI (Timepoint_6)",
}

# Clinical columns used to derive binary treatment coding (0/1).
COL_INITIAL_CHEMO = "Initial Chemo Therapy"
COL_INITIAL_CHEMO_NAME = "Name of Initial Chemo Therapy"
COL_INITIAL_CHEMO_START = "Number of days from Diagnosis to Initial Chemo Therapy Start date"
COL_INITIAL_CHEMO_END = "Number of days from Diagnosis to Initial Chemo Therapy end date"
COL_RADIATION = "Radiation Therapy"
COL_RADIATION_START = "Number of days from Diagnosis to Radiation Therapy Start date"
COL_RADIATION_END = "Number of days from Diagnosis to Radiation Therapy end date"
COL_TREATMENT_AFTER_2ND_PROGRESSION = "Treatment started after 2nd progression"
COL_NEW_TREATMENT_START = "Days from Diagnosis to new treatment"
COL_ADDITIONAL_THERAPY = "Additional Therapy"
COL_ADDITIONAL_START = "Number of Days from Diagnosis to Starting Additional Therapy"
COL_ADDITIONAL_END = "Number of Days from Diagnosis to Complete Additional Therapy"
COL_ADDITIONAL2_THERAPY = "2nd_Additional Therapy"
COL_ADDITIONAL2_START = "Number of Days from Diagnosis to Starting 2nd_Additional Therapy"
COL_ADDITIONAL2_END = "Number of Days from Dagnosis to Complete 2nd_Additional Therapy"
COL_IMMUNO_START = "Number of Days from Diagnosis to Start Immunotherapy"
COL_OTHER_START = "Number of Days from Diagnosis to Start Other Additional Therapy"

# File suffixes in MU-Glioma-Post timepoint folders.
MODALITY_SUFFIXES = {
    "t1": "_brain_t1n.nii.gz",
    "t1c": "_brain_t1c.nii.gz",
    "flair": "_brain_t2f.nii.gz",
    "t2": "_brain_t2w.nii.gz",
    "tumor_mask": "_tumorMask.nii.gz",
}


def _natural_key(text: str) -> List[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def _parse_timepoint_idx(name: str) -> Optional[int]:
    m = re.search(r"timepoint[_\- ]?(\d+)", name, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def _to_float_or_nan(v: object) -> float:
    if v is None:
        return float("nan")
    s = str(v).strip()
    if s == "" or s.lower() in {"na", "nan", "none", "n/a"}:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _is_yes(v: object) -> bool:
    s = str(v).strip().lower()
    return s in {"yes", "y", "true", "1"}


def _nonzero_norm_image(image: np.ndarray, clip_percent: float = 0.1) -> np.ndarray:
    """
    Match preprocessing behavior in data/preproc_prepare_data.py.
    """
    assert 0 <= clip_percent <= 0.5, f"clip_percent must be in [0, 0.5], got {clip_percent}"

    image = image.astype(np.float32, copy=False)
    nz_mask = image > 0
    if image[nz_mask].size == 0:
        return image

    if clip_percent > 0:
        minval = np.percentile(image[nz_mask], clip_percent)
        maxval = np.percentile(image[nz_mask], 100 - clip_percent)
        image[nz_mask & (image < minval)] = minval
        image[nz_mask & (image > maxval)] = maxval

    y = image[nz_mask]
    image_mean = np.mean(y)
    image_std = np.std(y)
    assert image_std != 0.0, f"Image std is zero: {image_std}"

    image = (image - image_mean) / image_std
    image = (image - image.min()) / (image.max() - image.min())
    return image.astype(np.float32)


def _read_nii(
    path: Path,
    orientation_axcode: str = "PLI",
    normalize_nonzero: bool = False,
    clip_percent: float = 0.1,
) -> np.ndarray:
    """
    Match preprocessing behavior in data/preproc_prepare_data.py.
    """
    img = nib.load(str(path))
    # Reorient to requested axcode (equivalent intent to reorient_nii.reorient).
    src_ornt = nib.orientations.io_orientation(img.affine)
    dst_ornt = nib.orientations.axcodes2ornt(tuple(orientation_axcode))
    transform = nib.orientations.ornt_transform(src_ornt, dst_ornt)
    data = nib.orientations.apply_orientation(img.get_fdata(), transform)
    new_aff = img.affine @ nib.orientations.inv_ornt_aff(transform, img.shape)
    img = nib.Nifti1Image(data, new_aff, img.header)
    arr = img.get_fdata().astype(np.float32)
    if normalize_nonzero:
        arr = _nonzero_norm_image(arr, clip_percent=clip_percent)
    return arr


def _find_file_by_suffix(folder: Path, suffix: str) -> Optional[Path]:
    for p in folder.iterdir():
        if p.is_file() and p.name.endswith(suffix):
            return p
    return None


def _load_xlsx_sheet_rows(xlsx_path: Path, sheet_name: str) -> List[Dict[str, str]]:
    """
    Read xlsx rows using stdlib zip/xml (no openpyxl dependency).
    """
    ns_main = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

    with zipfile.ZipFile(xlsx_path) as zf:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {r.attrib["Id"]: r.attrib["Target"] for r in rels}

        sheet_target = None
        for s in wb.findall("a:sheets/a:sheet", ns_main):
            if s.attrib.get("name") == sheet_name:
                rid = s.attrib[rel_ns]
                sheet_target = rel_map[rid]
                break
        if sheet_target is None:
            raise ValueError(f"Sheet '{sheet_name}' not found in {xlsx_path}")

        shared: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sst.findall("a:si", ns_main):
                text = "".join(t.text or "" for t in si.findall(".//a:t", ns_main))
                shared.append(text)

        def cell_text(cell: ET.Element) -> str:
            typ = cell.attrib.get("t")
            v = cell.find("a:v", ns_main)
            if v is None or v.text is None:
                return ""
            raw = v.text
            if typ == "s":
                idx = int(raw)
                if 0 <= idx < len(shared):
                    return shared[idx]
            return raw

        xml_path = "xl/" + sheet_target if not sheet_target.startswith("xl/") else sheet_target
        root = ET.fromstring(zf.read(xml_path))
        rows = root.findall(".//a:sheetData/a:row", ns_main)
        if not rows:
            return []

        headers = [cell_text(c).strip() for c in rows[0].findall("a:c", ns_main)]
        out: List[Dict[str, str]] = []
        for r in rows[1:]:
            vals = [cell_text(c).strip() for c in r.findall("a:c", ns_main)]
            if not any(vals):
                continue
            # pad short rows
            if len(vals) < len(headers):
                vals.extend([""] * (len(headers) - len(vals)))
            out.append({h: vals[i] for i, h in enumerate(headers)})
        return out


def _contains_tmz(text: object) -> bool:
    s = str(text).strip().lower()
    return ("temozolomide" in s) or ("temodar" in s)


def _to_int_or_none(v: object) -> Optional[int]:
    x = _to_float_or_nan(v)
    if np.isfinite(x):
        return int(round(float(x)))
    return None


def _add_interval(intervals: List[Tuple[int, int]], start: Optional[int], end: Optional[int]) -> None:
    if start is None and end is None:
        return
    if start is None:
        start = end
    if end is None:
        end = start
    if start is None or end is None:
        return
    lo = min(start, end)
    hi = max(start, end)
    intervals.append((int(lo), int(hi)))


def _derive_treatment_vector_from_row(row: Dict[str, str], abs_days: Sequence[int]) -> np.ndarray:
    """
    Day-specific treatment coding from clinical sheet, collapsed to SAILOR-style binary:
      0 = Chemoradiotherapy-era (CRT) / non-TMZ
      1 = Temozolomide-era (TMZ)

    Priority on each MRI day:
      1) If day is within CRT interval -> 0
      2) Else if day is within TMZ interval -> 1
      3) Else if day > CRT_end and patient has TMZ evidence -> 1 (adjuvant fallback)
      4) Else -> 0
    """
    crt_intervals: List[Tuple[int, int]] = []
    tmz_intervals: List[Tuple[int, int]] = []

    chemo_start = _to_int_or_none(row.get(COL_INITIAL_CHEMO_START, ""))
    chemo_end = _to_int_or_none(row.get(COL_INITIAL_CHEMO_END, ""))
    rt_start = _to_int_or_none(row.get(COL_RADIATION_START, ""))
    rt_end = _to_int_or_none(row.get(COL_RADIATION_END, ""))

    has_initial_chemo = _is_yes(row.get(COL_INITIAL_CHEMO, ""))
    has_radiation = _is_yes(row.get(COL_RADIATION, ""))
    initial_chemo_name = row.get(COL_INITIAL_CHEMO_NAME, "")

    # CRT interval: overlap-era represented by radiation +/- concurrent initial chemo.
    if has_radiation or has_initial_chemo:
        starts = [x for x in (rt_start, chemo_start) if x is not None]
        ends = [x for x in (rt_end, chemo_end) if x is not None]
        _add_interval(crt_intervals, min(starts) if starts else None, max(ends) if ends else None)

    # TMZ intervals: any explicit TMZ-bearing therapy windows.
    if has_initial_chemo and _contains_tmz(initial_chemo_name):
        _add_interval(tmz_intervals, chemo_start, chemo_end)

    add1_name = row.get(COL_ADDITIONAL_THERAPY, "")
    add1_start = _to_int_or_none(row.get(COL_ADDITIONAL_START, ""))
    add1_end = _to_int_or_none(row.get(COL_ADDITIONAL_END, ""))
    if _contains_tmz(add1_name):
        _add_interval(tmz_intervals, add1_start, add1_end)

    add2_name = row.get(COL_ADDITIONAL2_THERAPY, "")
    add2_start = _to_int_or_none(row.get(COL_ADDITIONAL2_START, ""))
    add2_end = _to_int_or_none(row.get(COL_ADDITIONAL2_END, ""))
    if _contains_tmz(add2_name):
        _add_interval(tmz_intervals, add2_start, add2_end)

    post2_name = row.get(COL_TREATMENT_AFTER_2ND_PROGRESSION, "")
    post2_start = _to_int_or_none(row.get(COL_NEW_TREATMENT_START, ""))
    if _contains_tmz(post2_name):
        _add_interval(tmz_intervals, post2_start, None)

    # Determine if patient ever has TMZ evidence in sheet.
    has_any_tmz_evidence = bool(tmz_intervals) or _contains_tmz(initial_chemo_name)
    crt_end = max((hi for _, hi in crt_intervals), default=None)

    treatment: List[int] = []
    for d in abs_days:
        day = int(d)
        in_crt = any(lo <= day <= hi for lo, hi in crt_intervals)
        if in_crt:
            treatment.append(0)
            continue

        in_tmz = any(lo <= day <= hi for lo, hi in tmz_intervals)
        if in_tmz:
            treatment.append(1)
            continue

        if has_any_tmz_evidence and crt_end is not None and day > crt_end:
            treatment.append(1)
        else:
            treatment.append(0)
    return np.array(treatment, dtype=np.int32)


def _build_patient_clinical_map(xlsx_path: Path) -> Dict[str, Dict[str, object]]:
    """
    Returns:
      {
        "PatientID_xxxx": {
           "tp_days": {1: day_tp1, 2: day_tp2, ...},
           "row": Dict[str, str],
        }
      }
    """
    rows = _load_xlsx_sheet_rows(xlsx_path, CLINICAL_SHEET_NAME)
    out: Dict[str, Dict[str, object]] = {}

    for row in rows:
        pid = row.get("Patient_ID", "").strip()
        if not pid:
            continue
        tp_days: Dict[int, int] = {}
        for tp, col in TIMEPOINT_DAY_COLUMNS.items():
            val = _to_float_or_nan(row.get(col, ""))
            if np.isfinite(val):
                tp_days[tp] = int(round(float(val)))
        out[pid] = {
            "tp_days": tp_days,
            "row": row,
        }
    return out


def _collect_patient_arrays(
    patient_dir: Path,
    day_map: Dict[int, int],
    clinical_row: Dict[str, str],
    clip_percent: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    session_dirs = [p for p in patient_dir.iterdir() if p.is_dir()]
    session_dirs.sort(key=lambda p: _natural_key(p.name))
    if not session_dirs:
        raise RuntimeError(f"{patient_dir.name}: no timepoint directories found.")

    all_modalities: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    abs_days: List[int] = []

    for sdir in session_dirs:
        tp = _parse_timepoint_idx(sdir.name)
        if tp is None:
            print(f"[Skip session] {patient_dir.name}/{sdir.name}: cannot parse timepoint index")
            continue

        t1 = _find_file_by_suffix(sdir, MODALITY_SUFFIXES["t1"])
        t1c = _find_file_by_suffix(sdir, MODALITY_SUFFIXES["t1c"])
        flair = _find_file_by_suffix(sdir, MODALITY_SUFFIXES["flair"])
        t2 = _find_file_by_suffix(sdir, MODALITY_SUFFIXES["t2"])
        mask = _find_file_by_suffix(sdir, MODALITY_SUFFIXES["tumor_mask"])
        missing = [
            k
            for k, v in [("t1", t1), ("t1c", t1c), ("flair", flair), ("t2", t2), ("tumor_mask", mask)]
            if v is None
        ]
        if missing:
            print(f"[Skip session] {patient_dir.name}/{sdir.name}: missing {missing}")
            continue

        day = day_map.get(tp, None)
        if day is None:
            print(f"[Skip session] {patient_dir.name}/{sdir.name}: no day in clinical xlsx")
            continue

        imgs = [
            _read_nii(t1, normalize_nonzero=True, clip_percent=clip_percent),
            _read_nii(t1c, normalize_nonzero=True, clip_percent=clip_percent),
            _read_nii(flair, normalize_nonzero=True, clip_percent=clip_percent),
            _read_nii(t2, normalize_nonzero=True, clip_percent=clip_percent),
        ]
        m = _read_nii(mask, normalize_nonzero=False)

        shape0 = imgs[0].shape
        for arr in imgs[1:] + [m]:
            if arr.shape != shape0:
                raise ValueError(
                    f"{patient_dir.name}/{sdir.name}: shape mismatch expected {shape0}, got {arr.shape}"
                )

        merged = np.zeros_like(m, dtype=np.int8)
        merged[m > 0] = ENHANCING

        all_modalities.extend(imgs)
        labels.append(merged)
        abs_days.append(int(day))

    if not labels:
        raise RuntimeError(f"{patient_dir.name}: no valid sessions after checks.")

    image = np.stack(all_modalities, axis=0).astype(np.float32)  # (4*T,H,W,D)
    label = np.stack(labels, axis=0).astype(np.int8)             # (T,H,W,D)

    first_day = abs_days[0]
    rel_days = np.array([max(int(round(d - first_day)), 0) for d in abs_days], dtype=np.int32)
    treatment = _derive_treatment_vector_from_row(row=clinical_row, abs_days=abs_days)
    return image, label, rel_days, treatment


def preprocess_mu_glioma_post(
    raw_root: Path,
    clinical_xlsx: Path,
    output_root: Path,
    patient_ids: Optional[Iterable[str]],
    clip_percent: float,
) -> None:
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")
    if not clinical_xlsx.exists():
        raise FileNotFoundError(f"clinical_xlsx not found: {clinical_xlsx}")

    output_root.mkdir(parents=True, exist_ok=True)
    patient_clinical_map = _build_patient_clinical_map(clinical_xlsx)

    patient_dirs = [p for p in raw_root.iterdir() if p.is_dir()]
    patient_dirs.sort(key=lambda p: _natural_key(p.name))
    selected = set(patient_ids) if patient_ids else None
    if selected is not None:
        patient_dirs = [p for p in patient_dirs if p.name in selected]
    if not patient_dirs:
        raise RuntimeError("No patient folders found.")

    for pdir in patient_dirs:
        pid = pdir.name
        if pid not in patient_clinical_map:
            print(f"[Skip] {pid}: not found in clinical xlsx sheet '{CLINICAL_SHEET_NAME}'.")
            continue
        try:
            clinical = patient_clinical_map[pid]
            image, label, days, treatment = _collect_patient_arrays(
                patient_dir=pdir,
                day_map=clinical["tp_days"],  # type: ignore[index]
                clinical_row=clinical["row"],  # type: ignore[index]
                clip_percent=clip_percent,
            )
        except Exception as e:
            print(f"[Skip] {pid}: {e}")
            continue

        np.save(output_root / f"{pid}_image.npy", image)
        np.save(output_root / f"{pid}_label.npy", label)
        np.save(output_root / f"{pid}_days.npy", days)
        np.save(output_root / f"{pid}_treatment.npy", treatment)
        print(
            f"[Saved] {pid} | image={image.shape} label={label.shape} "
            f"days={days.shape} treatment={treatment.shape}"
        )


def _parse_patient_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None or not s.strip():
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess MU-Glioma-Post + clinical xlsx to TaDiff npy format."
    )
    parser.add_argument("--raw_root", type=str, default="./dataset/MU-Glioma-Post")
    parser.add_argument(
        "--clinical_xlsx",
        type=str,
        default="./dataset/MU-Glioma-Post_ClinicalData-FINAL032025.xlsx",
    )
    parser.add_argument("--output_root", type=str, default="./mu_glioma_post_output")
    parser.add_argument("--patient_ids", type=str, default=None, help="Comma-separated patient IDs (optional).")
    parser.add_argument("--clip_percent", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_mu_glioma_post(
        raw_root=Path(args.raw_root),
        clinical_xlsx=Path(args.clinical_xlsx),
        output_root=Path(args.output_root),
        patient_ids=_parse_patient_list(args.patient_ids),
        clip_percent=float(args.clip_percent),
    )
