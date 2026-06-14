"""
Adaptive Excel loading utilities for the Sales Pipeline Analyzer.

This module detects the most likely data sheet, header row, and source-to-
canonical column mapping before the existing analysis pipeline runs.
"""

from __future__ import annotations

import io
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from processing import ALIASES, CANONICAL_COLS, LOB_MAPPING


HEADER_SCAN_ROWS = 30
PREVIEW_ROWS = 80
MIN_CONFIDENCE = 0.58

EXTRA_ALIASES: Dict[str, List[str]] = {
    "Opportunity Name": [
        "nama opportunity",
        "nama peluang",
        "opportunity id",
        "opportunity no",
        "project",
        "project name",
        "deal",
        "deal name",
    ],
    "Account Name": [
        "nama account",
        "nama akun",
        "pelanggan",
        "customer name",
        "client",
        "client name",
        "company name",
    ],
    "Stage": [
        "tahap",
        "fase",
        "pipeline stage",
        "sales stage",
        "opportunity stage",
        "status opportunity",
    ],
    "Schedule Amount": [
        "nilai",
        "nilai kontrak",
        "nilai project",
        "revenue",
        "total revenue",
        "sales amount",
        "pipeline amount",
        "contract amount",
        "schedule cv",
        "total cv",
    ],
    "Schedule Date": [
        "tanggal schedule",
        "tgl schedule",
        "forecast",
        "forecast close",
        "schedule",
        "target date",
    ],
    "Created Date": [
        "tanggal dibuat",
        "tgl dibuat",
        "create date",
        "created on",
        "submitted date",
    ],
    "Close Date": [
        "tanggal close",
        "tgl close",
        "tanggal closing",
        "tgl closing",
        "closed date",
        "target close date",
        "expected close date",
    ],
    "Last Stage Change Date": [
        "tanggal perubahan stage",
        "tgl perubahan stage",
        "stage change date",
        "last stage date",
    ],
    "Last Modified Date": [
        "tanggal update",
        "tgl update",
        "updated date",
        "updated at",
        "last update",
    ],
    "Industry Segment": [
        "segmen industri",
        "industry",
        "sector",
        "vertical",
        "business segment",
        "line business",
    ],
    "Pilar": [
        "pillar",
        "product",
        "product category",
        "kategori produk",
        "produk",
        "solution pillar",
    ],
    "AM Name": [
        "nama am",
        "account manager name",
        "sales",
        "sales name",
        "salesperson",
        "sales person",
    ],
    "Amount": [
        "nilai",
        "nilai kontrak",
        "revenue",
        "contract value",
        "sales amount",
    ],
    "Opportunity Owner": [
        "owner opportunity",
        "opportunity pic",
        "se name",
        "nama se",
        "sales engineer name",
        "pic",
    ],
    "Close Date (Year)": [
        "tahun close",
        "tahun closing",
        "closed year",
        "closing year",
    ],
}

CORE_COLUMNS = {
    "Opportunity Name",
    "Stage",
    "Schedule Amount",
    "Created Date",
    "Close Date",
    "Industry Segment",
}


def load_excel_adaptive(file_bytes: bytes) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load an Excel workbook by detecting the best sheet and table header."""
    workbook = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
    candidates: List[Dict[str, object]] = []

    for sheet_name in workbook.sheet_names:
        raw = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=sheet_name,
            header=None,
            nrows=PREVIEW_ROWS,
            engine="openpyxl",
            dtype=object,
        )
        if raw.dropna(how="all").empty:
            continue

        header_row, header_score, header_matches = detect_header_row(raw)
        df = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=sheet_name,
            header=header_row,
            engine="openpyxl",
            dtype=object,
        )
        df = clean_loaded_frame(df)
        mapping, confidence = smart_auto_map_columns(df)
        mapped_core = [c for c in CORE_COLUMNS if mapping.get(c)]
        mapped_any = [c for c in CANONICAL_COLS if mapping.get(c)]
        row_score = min(len(df) / 200.0, 1.0)
        col_score = min(len(df.columns) / 20.0, 1.0)
        total_score = (
            header_score
            + (len(mapped_core) * 2.0)
            + (len(mapped_any) * 0.35)
            + row_score
            + col_score
        )

        candidates.append(
            {
                "sheet_name": sheet_name,
                "header_row": header_row,
                "header_score": header_score,
                "header_matches": header_matches,
                "mapping": mapping,
                "confidence": confidence,
                "rows": len(df),
                "columns": len(df.columns),
                "mapped_core": mapped_core,
                "mapped_any": mapped_any,
                "score": total_score,
                "df": df,
            }
        )

    if not candidates:
        raise ValueError("Workbook tidak memiliki sheet berisi data yang bisa dibaca.")

    best = max(candidates, key=lambda item: item["score"])
    diagnostics = {
        key: value
        for key, value in best.items()
        if key != "df"
    }
    diagnostics["sheet_candidates"] = [
        {
            "sheet_name": c["sheet_name"],
            "header_row": c["header_row"],
            "rows": c["rows"],
            "columns": c["columns"],
            "mapped_columns": len(c["mapped_any"]),
            "mapped_core": len(c["mapped_core"]),
            "score": round(float(c["score"]), 2),
        }
        for c in sorted(candidates, key=lambda item: item["score"], reverse=True)
    ]
    return best["df"], diagnostics


def detect_header_row(raw: pd.DataFrame) -> Tuple[int, float, List[str]]:
    """Find the row that most likely contains table headers."""
    best_row = 0
    best_score = -1.0
    best_matches: List[str] = []
    max_rows = min(len(raw), HEADER_SCAN_ROWS)

    for row_idx in range(max_rows):
        values = [value for value in raw.iloc[row_idx].tolist() if not _is_blank(value)]
        if len(values) < 2:
            continue

        labels = [_stringify(value) for value in values]
        non_numeric_ratio = np.mean([not _looks_numeric(label) for label in labels])
        unique_ratio = len(set(normalize_label(label) for label in labels)) / max(len(labels), 1)

        matches = []
        label_scores = []
        for label in labels:
            canon, score = best_label_match(label)
            label_scores.append(score)
            if canon and score >= 0.72:
                matches.append(canon)

        strong_count = len(set(matches))
        score = (
            sum(label_scores)
            + (strong_count * 1.5)
            + (len(labels) * 0.08)
            + (non_numeric_ratio * 0.7)
            + (unique_ratio * 0.5)
        )

        if score > best_score:
            best_row = row_idx
            best_score = score
            best_matches = sorted(set(matches))

    return best_row, round(float(max(best_score, 0.0)), 2), best_matches


def clean_loaded_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean empty rows/columns and normalize column labels after reading."""
    out = df.copy()
    out = out.dropna(how="all").dropna(axis=1, how="all")

    columns: List[str] = []
    counts: Dict[str, int] = {}
    for idx, col in enumerate(out.columns):
        label = _stringify(col).strip()
        if not label or label.lower().startswith("unnamed:"):
            label = f"Column {idx + 1}"
        label = re.sub(r"\s+", " ", label)
        counts[label] = counts.get(label, 0) + 1
        if counts[label] > 1:
            label = f"{label}.{counts[label]}"
        columns.append(label)

    out.columns = columns
    return out.reset_index(drop=True)


def smart_auto_map_columns(df: pd.DataFrame) -> Tuple[Dict[str, Optional[str]], Dict[str, float]]:
    """Map source columns to canonical names using labels and data profiles."""
    mapping: Dict[str, Optional[str]] = {canon: None for canon in CANONICAL_COLS}
    confidence: Dict[str, float] = {canon: 0.0 for canon in CANONICAL_COLS}
    used_sources = set()

    for canon in CANONICAL_COLS:
        scored: List[Tuple[float, str]] = []
        for source in df.columns:
            if source in used_sources:
                continue
            label_score = label_match_score(canon, str(source))
            content_score = column_content_score(canon, df[source])
            score = (label_score * 0.72) + (content_score * 0.28)
            scored.append((score, source))

        if not scored:
            continue

        best_score, best_source = max(scored, key=lambda item: item[0])
        if best_score >= MIN_CONFIDENCE:
            mapping[canon] = best_source
            confidence[canon] = round(float(best_score), 2)
            used_sources.add(best_source)

    return mapping, confidence


def best_label_match(label: str) -> Tuple[Optional[str], float]:
    """Return the canonical column that best matches a raw header label."""
    scored = [(label_match_score(canon, label), canon) for canon in CANONICAL_COLS]
    score, canon = max(scored, key=lambda item: item[0])
    return (canon, score) if score > 0 else (None, 0.0)


def label_match_score(canon: str, label: str) -> float:
    """Score how closely a source label matches one canonical column."""
    source = normalize_label(label)
    if not source:
        return 0.0

    aliases = [canon] + ALIASES.get(canon, []) + EXTRA_ALIASES.get(canon, [])
    scores = []
    for alias in aliases:
        target = normalize_label(alias)
        if not target:
            continue
        if source == target:
            scores.append(1.0)
        elif source in target or target in source:
            shorter = min(len(source), len(target))
            longer = max(len(source), len(target))
            scores.append(0.82 + (0.12 * shorter / max(longer, 1)))
        else:
            scores.append(SequenceMatcher(None, source, target).ratio())
    return float(max(scores or [0.0]))


def column_content_score(canon: str, series: pd.Series) -> float:
    """Score a source column based on sample values."""
    sample = series.dropna().head(100)
    if sample.empty:
        return 0.0

    if canon in {"Created Date", "Close Date", "Schedule Date", "Last Modified Date", "Last Stage Change Date"}:
        return _date_ratio(sample)

    if canon == "Close Date (Year)":
        numeric = pd.to_numeric(sample, errors="coerce")
        valid = numeric.between(1990, 2100).mean()
        return float(valid)

    if canon in {"Schedule Amount", "Amount"}:
        numeric = _coerce_numeric(sample)
        numeric_ratio = numeric.notna().mean()
        large_value_ratio = (numeric.abs() >= 1000).mean() if numeric.notna().any() else 0
        return float((numeric_ratio * 0.75) + (large_value_ratio * 0.25))

    text = sample.astype(str).str.strip().str.lower()

    if canon == "Stage":
        stage_terms = (
            "closed won",
            "closed lost",
            "won",
            "lost",
            "open",
            "proposal",
            "negotiation",
            "qualification",
            "prospect",
        )
        return float(text.apply(lambda value: any(term in value for term in stage_terms)).mean())

    if canon == "Industry Segment":
        lob_keys = set(LOB_MAPPING.keys())
        direct = text.isin(lob_keys).mean()
        sector_terms = ("industry", "manufacturing", "financial", "retail", "government", "energy", "oil", "ict")
        semantic = text.apply(lambda value: any(term in value for term in sector_terms)).mean()
        return float(max(direct, semantic * 0.8))

    if canon == "Pilar":
        product_terms = ("connectivity", "iot", "solution", "cloud", "security", "fleet", "product", "pilar")
        return float(text.apply(lambda value: any(term in value for term in product_terms)).mean())

    if canon in {"Opportunity Name", "Account Name", "AM Name", "Opportunity Owner"}:
        avg_len = text.str.len().clip(upper=60).mean() / 60.0
        mostly_text = (pd.to_numeric(sample, errors="coerce").isna()).mean()
        return float((avg_len * 0.35) + (mostly_text * 0.65))

    return 0.0


def normalize_label(value: object) -> str:
    """Normalize labels for fuzzy matching."""
    text = _stringify(value).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _date_ratio(sample: pd.Series) -> float:
    numeric = pd.to_numeric(sample, errors="coerce")
    numeric_ratio = numeric.notna().mean()
    if numeric_ratio >= 0.8:
        excel_serial_ratio = numeric.between(20000, 80000).mean()
        year_ratio = numeric.between(1990, 2100).mean()
        if max(excel_serial_ratio, year_ratio) < 0.5:
            return 0.0

    parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
    return float(parsed.notna().mean())


def _coerce_numeric(sample: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(sample):
        return pd.to_numeric(sample, errors="coerce")
    cleaned = (
        sample.astype(str)
        .str.replace(r"[^0-9,.\-]", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _looks_numeric(value: str) -> bool:
    return pd.notna(pd.to_numeric(value, errors="coerce"))


def _is_blank(value: object) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _stringify(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)
