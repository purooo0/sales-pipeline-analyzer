"""
Streamlit UI for the Sales Pipeline Analyzer.

Flow:
1) Upload Excel -> 2) Show columns -> 3) Select columns -> 4) Map to canonical -> 5) Run analyses

Visual theme uses a single Telkomsel red accent for a professional, consistent look.
"""
import hashlib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from excel_adapter import load_excel_adaptive
from processing import (
    CANONICAL_COLS,
    CHART_GRID,
    CHART_MUTED,
    CHART_PALETTE,
    CHART_TEXT,
    auto_map_columns,
    preprocess,
    run_bagian1,
    run_bagian2,
    idr_label,
    LOB_MAPPING,
)

# ---- Theme constants ----
TELKOMSEL_RED: str = "#e60000"
NEUTRAL_GRAY: str = "#d9d9d9"

def _chart_palette(n: int) -> List[str]:
    """Return enough colors for n chart categories."""
    if n <= 0:
        return []
    repeats = int((n + len(CHART_PALETTE) - 1) / len(CHART_PALETTE))
    return (CHART_PALETTE * repeats)[:n]

# ---- Helpers for UI rendering (defined early to avoid NameError) ----
def _render_top_am_section(df_am: pd.DataFrame, metric: str, xlabel: str, key_suffix: str):
    """Render the Top 5 AM controls and per-LoB tabs for a given metric.

    Parameters
    ----------
    df_am : pd.DataFrame
        Aggregated AM performance dataframe produced by `run_bagian2()`.
    metric : str
        One of {"conversion_rate", "total_cv", "total"}.
    xlabel : str
        X-axis label for the bars.
    key_suffix : str
        Suffix for Streamlit widget keys to avoid collisions.
    """
    # Controls: pick LoB(s), years, Top N
    lob_options = sorted(df_am["LoB"].dropna().unique().tolist())
    c1, c2, c3, c4 = st.columns([2, 2, 1, 2])
    with c1:
        lob_single = st.selectbox("LoB", options=lob_options, key=f"lob_{key_suffix}")
    # union of years for all lobs
    years_all = sorted(df_am["Year"].dropna().unique().tolist())
    with c2:
        years_sel = st.multiselect("Years", options=years_all, default=years_all, key=f"years_{key_suffix}")
    with c3:
        topn = st.slider("Top N", min_value=3, max_value=10, value=5, step=1, key=f"topn_{key_suffix}")
    with c4:
        show_all = st.checkbox("Tampilkan semua LoB", value=False, key=f"all_{key_suffix}")

    if not years_sel:
        st.info("Pilih minimal 1 tahun.")
        return

    lobs_to_show = lob_options if show_all else [lob_single]
    for lob in lobs_to_show:
        st.markdown(f"#### {lob}")
        _render_top_am_for_lob(df_am, lob, years_sel, topn, metric, xlabel)

def _render_top_am_for_lob(df_am: pd.DataFrame, lob: str, years_sel: List[int], topn: int, metric: str, xlabel: str):
    """Render Top N AM bars for a single LoB across selected years."""
    tabs = st.tabs([str(y) for y in years_sel])
    for i, y in enumerate(years_sel):
        with tabs[i]:
            sub = df_am[(df_am["LoB"] == lob) & (df_am["Year"] == y)].copy()
            if sub.empty:
                st.warning("Tidak ada data untuk kombinasi ini.")
                continue
            sub = sub.sort_values(by=metric, ascending=False).head(topn)
            sub = sub.sort_values(by=metric, ascending=True)
            fig, ax = plt.subplots(figsize=(10, max(3.8, 0.5 * len(sub) + 2)))
            ax.barh(
                sub["am"],
                sub[metric],
                color=_chart_palette(len(sub)),
                edgecolor="white",
                linewidth=1,
            )
            # Labels on bars
            max_value = sub[metric].dropna().max()
            offset = float(max_value) * 0.015 if pd.notna(max_value) and max_value else 0.5
            for j, v in enumerate(sub[metric].tolist()):
                if pd.isna(v):
                    continue
                label = _fmt_metric(v, metric)
                ax.text(v + offset, j, label, va="center", ha="left", fontsize=9, color=CHART_TEXT)
            ax.set_xlabel(xlabel, color=CHART_MUTED, labelpad=10)
            ax.set_ylabel("Account Manager", color=CHART_MUTED, labelpad=10)
            ax.set_title(f"Top {len(sub)} AM - {y}", loc="left", fontsize=14, fontweight="700", color=CHART_TEXT, pad=14)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color(CHART_GRID)
            ax.spines["bottom"].set_color(CHART_GRID)
            ax.grid(axis="x", color=CHART_GRID, linestyle="-", linewidth=0.8)
            ax.grid(False, axis="y")
            ax.tick_params(axis="x", colors=CHART_MUTED)
            ax.tick_params(axis="y", colors=CHART_TEXT)
            if pd.notna(max_value) and max_value > 0:
                ax.set_xlim(right=max_value * 1.18)
            fig.tight_layout()
            st.pyplot(fig)

def _fmt_metric(v, metric: str) -> str:
    """Format metric values for in-bar labels."""
    if pd.isna(v):
        return ""
    if metric == "conversion_rate":
        return f"{v:.1f}%"
    if metric == "total_cv":
        return idr_label(v)
    if metric == "total":
        return f"{int(v)}"
    return str(v)

# Simple professional table styling with Telkomsel red accent (minimal colors)
def style_df(df: pd.DataFrame):
    """Return a styled dataframe with subtle Telkomsel-red accents (no gradients)."""
    if df is None or df.empty:
        return df
    styler = df.style.set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#ffe5e5"),
            ("color", "#7f0000"),
            ("font-weight", "600"),
            ("border-bottom", "1px solid #f0b3b3"),
        ]},
        {"selector": "td", "props": [
            ("border-bottom", "1px solid #f0f0f0"),
        ]},
    ]).set_properties(**{"text-align": "left"})
    return styler

st.set_page_config(page_title="Sales Pipeline Analyzer", layout="wide")

st.title("Sales Pipeline Analyzer")

st.sidebar.header("1) Upload Excel")
uploaded = st.sidebar.file_uploader("Upload .xlsx", type=["xlsx"])

@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes):
    """Load an Excel file using adaptive sheet/header detection."""
    return load_excel_adaptive(file_bytes)

def dependency_sets() -> Dict[str, List[str]]:
    """Return minimal per-output dependencies to drive warnings/skips."""
    return {
        # Bagian 1
        "b1_summary": ["Industry Segment", "Schedule Amount", "Stage", "Opportunity Name", "Created Date", "Close Date"],
        "b1_pie": ["Pilar", "Schedule Amount", "Stage"],
        "b1_quarter_bars": ["Schedule Date", "Schedule Amount", "Stage"],
        "b1_pivot": ["Schedule Date", "Product Type", "Stage", "Opportunity Name"],
        "b1_top_bottom": ["Close Date (Year)", "Opportunity Name", "Account Name", "Close Date", "Schedule Amount", "Stage"],
        "b1_open_pipeline": ["Stage", "Close Date", "Schedule Amount", "Opportunity Name", "Account Name"],
        # Bagian 2
        "b2_core": ["Stage", "Created Date", "Close Date", "Industry Segment"],
        "b2_stage_close": ["Last Stage Change Date", "Close Date", "Pilar", "Industry Segment"],
        "b2_se": ["Last Stage Change Date", "Close Date", "Opportunity Owner"],
        "b2_top5_am": ["Close Date", "AM Name", "Opportunity Name", "Stage", "Schedule Amount"],
    }

def missing_for(deps: List[str], available_cols: List[str]) -> List[str]:
    """Return missing dependency columns from a list of available columns."""
    return [c for c in deps if c not in available_cols]

# Step 2: read + show columns
if uploaded:
    file_bytes = uploaded.getvalue()
    file_signature = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get("file_signature") != file_signature:
        st.session_state["file_signature"] = file_signature
        st.session_state.pop("results", None)

    try:
        df_raw, excel_info = load_excel(file_bytes)
    except Exception as exc:
        st.error(f"File Excel tidak bisa dibaca otomatis: {exc}")
        st.stop()

    st.subheader("Preview Data")
    st.dataframe(df_raw.head(20), use_container_width=True)

    st.subheader("Excel Structure Detection")
    c_sheet, c_header, c_shape, c_mapped = st.columns(4)
    with c_sheet:
        st.metric("Sheet", str(excel_info.get("sheet_name", "-")))
    with c_header:
        header_display = int(excel_info.get("header_row", 0)) + 1
        st.metric("Header Row", header_display)
    with c_shape:
        st.metric("Rows x Columns", f"{excel_info.get('rows', 0)} x {excel_info.get('columns', 0)}")
    with c_mapped:
        st.metric("Mapped Fields", len(excel_info.get("mapped_any", [])))

    with st.expander("Automatic Mapping Details", expanded=False):
        confidence = excel_info.get("confidence", {})
        detected_mapping = excel_info.get("mapping", {})
        map_rows = []
        for canon in CANONICAL_COLS:
            source = detected_mapping.get(canon)
            if source:
                map_rows.append({
                    "Canonical Field": canon,
                    "Detected Column": source,
                    "Confidence": f"{confidence.get(canon, 0) * 100:.0f}%",
                })
        if map_rows:
            st.dataframe(pd.DataFrame(map_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada kolom yang bisa dipetakan otomatis.")

        candidates = excel_info.get("sheet_candidates", [])
        if candidates:
            st.markdown("#### Sheet Candidates")
            st.dataframe(pd.DataFrame(candidates), use_container_width=True, hide_index=True)

    st.subheader("Columns Detected")
    all_cols = list(df_raw.columns)
    st.write(", ".join(map(str, all_cols)))

    # Step 3: select columns to use
    st.sidebar.header("2) Select Columns to Use")
    selected_cols = st.sidebar.multiselect(
        "Pick columns to include for processing",
        options=all_cols,
        default=all_cols,
    )

    # Step 4: mapping UI
    st.sidebar.header("3) Column Mapping (only if needed)")
    detected_mapping = excel_info.get("mapping", {})
    fallback_map = auto_map_columns(selected_cols)
    auto_map = {}
    for canon in CANONICAL_COLS:
        detected_source = detected_mapping.get(canon)
        auto_map[canon] = detected_source if detected_source in selected_cols else fallback_map.get(canon)

    mapping: Dict[str, Optional[str]] = {}
    with st.sidebar.expander("Map to Canonical Names", expanded=False):
        for canon in CANONICAL_COLS:
            opts = [None] + selected_cols
            prefill = auto_map.get(canon)
            conf = excel_info.get("confidence", {}).get(canon, 0)
            label = f"{canon} ({conf * 100:.0f}% auto)" if prefill else canon
            choice = st.selectbox(label, options=opts, index=(opts.index(prefill) if prefill in opts else 0), key=f"map_{canon}")
            mapping[canon] = choice

    # Industry Segment filter (mapped to standardized 8 LoB)
    st.sidebar.header("4) Filters")
    seg_col = mapping.get("Industry Segment") or ("Industry Segment" if "Industry Segment" in df_raw.columns else None)
    selected_lobs: List[str] = []
    if seg_col:
        raw_seg = df_raw[seg_col].astype(str).str.strip().str.lower()
        mapped_lob = raw_seg.map(LOB_MAPPING)
        lob_options = sorted(pd.Series(mapped_lob).dropna().unique().tolist())
        selected_lobs = st.sidebar.multiselect("Industry Segment (mapped)", options=lob_options, default=lob_options)

    st.sidebar.header("5) Run (Per Fitur)")
    feature_options = [
        "Summary",
        "Product Mix Donuts",
        "Quarterly Bars",
        "Opportunity Count Table",
        "Top/Bottom 5 Closed Won",
        "Open Pipeline by Close Year",
        "Avg Sales Cycle per LoB",
        "Avg Stage->Close per LoB & Product",
        "Avg Stage->Close per SE",
        "Win Rate per LoB",
        "Top 5 AM: Conversion Rate",
        "Top 5 AM: Total CV",
        "Top 5 AM: Total Deals",
    ]
    selected_features = st.sidebar.multiselect(
        "Pilih fitur yang ingin ditampilkan",
        options=feature_options,
        default=["Summary", "Product Mix Donuts", "Quarterly Bars"],
    )
    btn_run = st.sidebar.button("Run")

    # Preprocess with mapping before running analyses
    # Keep only selected columns first
    df_sel = df_raw[selected_cols].copy()
    df = preprocess(df_sel, mapping)
    available = list(df.columns)

    # Dependency-driven notices
    deps = dependency_sets()

    # Run analyses and cache results when user clicks Run
    if btn_run:
        # Apply mapped LoB filter early so both bagian1 & bagian2 konsisten
        df_filtered = df.copy()
        if "Industry Segment" in df_filtered.columns and selected_lobs:
            seg_series = df_filtered["Industry Segment"].astype(str).str.strip().str.lower()
            lob_series = seg_series.map(LOB_MAPPING)
            df_filtered = df_filtered[lob_series.isin(selected_lobs)].copy()

        # bagian1 already supports industry_filter, but we filtered by mapped LoB above -> pass None to avoid double filtering
        res1 = run_bagian1(df_filtered, industry_filter=None)
        res2 = run_bagian2(df_filtered)
        st.session_state["results"] = {"res1": res1, "res2": res2}

    # Use cached results if available
    results = st.session_state.get("results")
    if not results:
        st.info("Klik Run untuk menjalankan analisis.")
        st.stop()

    res1 = results["res1"]
    res2 = results["res2"]

    # Show accumulated warnings up-front
    for w in res1.get("warnings", []) + res2.get("warnings", []):
        st.warning(w)

    # ----- Bagian 1 features -----
    if "Summary" in selected_features and "summary" in res1:
        st.markdown("### Summary")
        s = res1["summary"]
        st.write(f"Total Pipeline: {s['total_opps']} opps | CV IDR {s['total_cv_bn']} Bn")
        st.write(f"Closed Won: {s['won_opps']} opps | CV IDR {s['won_cv_bn']} Bn")
        st.write(f"Conversion Rate: {s['conversion_rate']}%")

    if "Product Mix Donuts" in selected_features and "figures" in res1:
        if "pie_all" in res1["figures"]:
            st.markdown("### Contract Value by Product Type")
            st.pyplot(res1["figures"]["pie_all"])
        if "pie_won" in res1["figures"]:
            st.markdown("### Won Contract Value by Product Type")
            st.pyplot(res1["figures"]["pie_won"])

    if "Quarterly Bars" in selected_features and "figures" in res1 and "bars_quarterly" in res1["figures"]:
        st.markdown("### CV per Quarter - All Stage vs Won Only")
        st.pyplot(res1["figures"]["bars_quarterly"])

    if "Opportunity Count Table" in selected_features and "pivot_table" in res1:
        st.markdown("### Opportunity Count Table (Year-Quarter by Product Type & Stage)")
        st.dataframe(style_df(res1["pivot_table"]), use_container_width=True)

    if "Top/Bottom 5 Closed Won" in selected_features:
        if "top5" in res1 and res1["top5"]:
            st.markdown("### Top 5 Closed Won Opportunities per Year")
            for y, t in res1["top5"].items():
                st.markdown(f"**Year {y}")
                st.dataframe(style_df(t), use_container_width=True)
        if "bottom5" in res1 and res1["bottom5"]:
            st.markdown("### Bottom 5 Closed Won Opportunities per Year")
            for y, t in res1["bottom5"].items():
                st.markdown(f"**Year {y}")
                st.dataframe(style_df(t), use_container_width=True)

    if "Open Pipeline by Close Year" in selected_features and "open_pipeline" in res1 and res1["open_pipeline"]:
        st.markdown("### Open Opportunities per Close Year")
        for y, t in res1["open_pipeline"].items():
            st.markdown(f"**Close Year {y}")
            st.dataframe(style_df(t), use_container_width=True)

    # ----- Bagian 2 features -----
    if "Avg Sales Cycle per LoB" in selected_features:
        if "tables" in res2 and "cycle_per_lob" in res2["tables"]:
            st.markdown("### Average Sales Cycle Time per LoB (table)")
            st.dataframe(style_df(res2["tables"]["cycle_per_lob"]), use_container_width=True)
        if "figures" in res2 and "cycle_per_lob" in res2["figures"]:
            st.pyplot(res2["figures"]["cycle_per_lob"])

    if "Avg Stage->Close per LoB & Product" in selected_features and "figures" in res2 and "stage_to_close_lob_product" in res2["figures"]:
        st.markdown("### Avg Stage to Close Time per LoB and Product Type")
        st.pyplot(res2["figures"]["stage_to_close_lob_product"])

    if "Avg Stage->Close per SE" in selected_features and "figures" in res2 and "stage_to_close_per_se" in res2["figures"]:
        st.markdown("### Avg Stage to Close Time per Opportunity Owner (SE)")
        st.pyplot(res2["figures"]["stage_to_close_per_se"])

    if "Win Rate per LoB" in selected_features:
        if "tables" in res2 and "win_rate_per_lob" in res2["tables"]:
            st.markdown("### Win Rate per LoB (table)")
            st.dataframe(style_df(res2["tables"]["win_rate_per_lob"]), use_container_width=True)
        if "figures" in res2 and "win_rate_per_lob" in res2["figures"]:
            st.pyplot(res2["figures"]["win_rate_per_lob"])

    if "Top 5 AM: Conversion Rate" in selected_features:
        st.markdown("### Top 5 AM: Conversion Rate")
        df_am = res2.get("tables", {}).get("top5_am_df")
        if df_am is not None and not df_am.empty:
            _render_top_am_section(df_am, metric="conversion_rate", xlabel="Conversion Rate (%)", key_suffix="conv")
        elif "figures" in res2 and "top5_am_conversion" in res2["figures"]:
            st.pyplot(res2["figures"]["top5_am_conversion"])

    if "Top 5 AM: Total CV" in selected_features:
        st.markdown("### Top 5 AM: Total CV")
        df_am = res2.get("tables", {}).get("top5_am_df")
        if df_am is not None and not df_am.empty:
            _render_top_am_section(df_am, metric="total_cv", xlabel="Total Contract Value (Rupiah)", key_suffix="cv")
        elif "figures" in res2 and "top5_am_total_cv" in res2["figures"]:
            st.pyplot(res2["figures"]["top5_am_total_cv"])

    if "Top 5 AM: Total Deals" in selected_features:
        st.markdown("### Top 5 AM: Total Deals")
        df_am = res2.get("tables", {}).get("top5_am_df")
        if df_am is not None and not df_am.empty:
            _render_top_am_section(df_am, metric="total", xlabel="Total Deals (Opportunity Count)", key_suffix="deals")
        elif "figures" in res2 and "top5_am_total" in res2["figures"]:
            st.pyplot(res2["figures"]["top5_am_total"])

else:
    st.info("Upload file Excel (.xlsx) untuk memulai.")
