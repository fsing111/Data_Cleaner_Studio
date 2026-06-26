"""
Automated Data Cleaning Studio — Streamlit App
================================================
Upload a messy CSV/Excel, choose cleaning options, preview before/after,
visualize data quality, mask sensitive data, and download the cleaned file.

Supports Chinese / English language switching (default: Chinese).

Run with:
    streamlit run app.py
"""

import io
import json

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import streamlit as st

from cleaning_engine import clean_dataframe, detect_sensitive_columns

# ── Matplotlib 中文字体设置 ────────────────────────────────────────────────
_CJK_FONTS = [
    "PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Microsoft YaHei",
    "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Arial Unicode MS",
]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
_chinese_font = None
for _font in _CJK_FONTS:
    if _font in _available_fonts:
        _chinese_font = _font
        break

if _chinese_font:
    plt.rcParams["font.family"] = _chinese_font
    plt.rcParams["axes.unicode_minus"] = False
else:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*")

# ══════════════════════════════════════════════════════════════════════════════
# 国际化翻译字典 / i18n Translation Dictionary
# ══════════════════════════════════════════════════════════════════════════════
T = {
    "zh": {
        # ── 页面框架 ──
        "page.title": "🧹 自动化数据清洗平台",
        "page.caption": "上传 CSV 或 Excel 文件 → 自动类型检测、去重、缺失值处理、异常值处理、数据脱敏 → 下载清洗后的数据集。",
        "page.divider": None,

        # ── 语言切换 ──
        "lang.zh": "中文",
        "lang.en": "English",
        "lang.label": "语言 / Language",

        # ── 未上传文件时的介绍 ──
        "intro.hint": "👆 上传一个 CSV 或 Excel 文件开始使用。本工具会自动检测列类型、清理文本、去重、处理缺失值、处理异常值，并可选择性脱敏敏感数据。",
        "intro.header": "功能概览",
        "intro.card1.title": "🔍 智能类型检测",
        "intro.card1.desc": "自动识别文本列中的数值（含货币/百分比格式）、日期时间、布尔值（是/否）、低基数分类字段，并转换为正确的数据类型。",
        "intro.card2.title": "🧼 数据清洗",
        "intro.card2.desc": "列名标准化为 snake_case 格式，去除首尾空格和多余空白，删除完全重复的行。",
        "intro.card3.title": "📊 缺失值与异常值",
        "intro.card3.desc": "根据列类型智能填充缺失值（中位数/众数/插值），基于 IQR 算法检测异常值，支持截断或删除。",
        "intro.card4.title": "🔒 数据脱敏",
        "intro.card4.desc": "自动识别并脱敏敏感数据（姓名、手机号、邮箱、身份证号、地址），保护隐私安全。",

        # ── 侧边栏 ──
        "sidebar.header": "⚙️ 清洗选项",
        "sidebar.missing_values": "缺失值处理",
        "sidebar.numeric_cols": "数值列填充策略",
        "sidebar.numeric_help": "选择填充缺失数值的方法。",
        "sidebar.numeric.median": "中位数",
        "sidebar.numeric.mean": "均值",
        "sidebar.numeric.zero": "填零",
        "sidebar.categorical_cols": "分类/文本列填充策略",
        "sidebar.categorical_help": "'众数'用出现最多的值填充；'缺失标签'用文字'Missing'填充。",
        "sidebar.categorical.mode": "众数",
        "sidebar.categorical.missing_label": "缺失标签",
        "sidebar.outliers": "异常值处理（数值列）",
        "sidebar.outlier_method": "异常值策略",
        "sidebar.outlier.cap": "截断到 IQR 边界（Winsorize）",
        "sidebar.outlier.remove": "删除含异常值的行",
        "sidebar.outlier.none": "仅检测 — 不做处理",
        "sidebar.iqr_help": "值越小 = 异常值检测越严格。1.5 是统计学标准惯例。",
        "sidebar.masking_header": "🔒 数据脱敏",
        "sidebar.masking_enable": "启用数据脱敏",
        "sidebar.masking_enable_help": "自动检测并脱敏敏感数据（姓名、手机号、邮箱、身份证号、地址）。",
        "sidebar.masking_caption": "选择要脱敏的数据类型：",
        "sidebar.masking.name": "姓名",
        "sidebar.masking.phone": "手机号",
        "sidebar.masking.email": "邮箱",
        "sidebar.masking.id_card": "身份证号",
        "sidebar.masking.address": "地址",
        "sidebar.pipeline_order": (
            "**处理流程：**\n"
            "1. 标准化列名\n"
            "2. 去除空白字符\n"
            "3. 移除重复行\n"
            "4. 智能类型检测与转换\n"
            "5. 缺失值处理\n"
            "6. 异常值处理\n"
            "7. 数据脱敏（如启用）"
        ),

        # ── 清洗进度 ──
        "progress.status": "🧹 正在清洗数据...",
        "progress.complete": "✅ 清洗完成！",
        "progress.detecting": "正在检测敏感列...",
        "progress.done": "完成！",
        "progress.step_fmt": "步骤 {step}/{total}: {name}...",
        "progress.step.0": "标准化列名",
        "progress.step.1": "文本规范化",
        "progress.step.2": "移除重复行",
        "progress.step.3": "智能类型检测与转换",
        "progress.step.4": "二次文本规范化",
        "progress.step.5": "缺失值处理",
        "progress.step.6": "异常值处理",
        "progress.step.7": "数据脱敏",

        # ── 摘要指标卡 ──
        "metrics.original_rows": "原始行数",
        "metrics.final_rows": "清洗后行数",
        "metrics.duplicates_removed": "已去重行数",
        "metrics.columns_retyped": "已转换类型列数",
        "metrics.quality_score": "质量评分",

        # ── Tab 标签 ──
        "tab.cleaned_preview": "🧾 清洗预览",
        "tab.quality_report": "📋 质量报告",
        "tab.visualizations": "📊 数据可视化",
        "tab.type_conversions": "🔄 类型转换",
        "tab.missing_values": "❓ 缺失值",
        "tab.outliers": "📈 异常值",
        "tab.masking": "🔒 脱敏",
        "tab.column_renames": "🏷️ 列名变更",

        # ── 清洗预览 Tab ──
        "preview.showing_rows": "显示前 20 行（共 {n} 行）。",

        # ── 质量报告 Tab ──
        "quality.header": "📋 数据质量报告",
        "quality.overall_score": "综合质量评分 / 100",
        "quality.scoring": "**评分规则：**",
        "quality.scoring_detail": "完整性 (50%) + 唯一性 (20%) + 类型一致性 (30%)",
        "quality.completeness_before": "完整性（清洗前）",
        "quality.completeness_after": "完整性（清洗后）",
        "quality.dup_rate_before": "重复率（清洗前）",
        "quality.dup_rate_after": "重复率（清洗后）",
        "quality.rows_removed": "移除行数",
        "quality.columns_retyped_label": "类型转换列数",
        "quality.numeric_stats": "数值列统计摘要（清洗后）",

        # ── 可视化 Tab ──
        "viz.missing_title": "缺失值对比",
        "viz.missing_before": "清洗前",
        "viz.missing_after": "清洗后",
        "viz.missing_xlabel": "缺失数量",
        "viz.missing_suptitle": "缺失值：清洗前 vs 清洗后",
        "viz.outlier_title": "异常值箱线图",
        "viz.outlier_suptitle": "异常值处理：清洗前 vs 清洗后（箱线图）",
        "viz.dist_title": "分布直方图",
        "viz.dist_suptitle": "分布变化：清洗前 vs 清洗后",
        "viz.no_outlier_cols": "没有需要可视化的异常值列。",

        # ── 类型转换 Tab ──
        "type.changed_header": "**✨ 已自动转换的列：**",
        "type.unchanged_expander": "未变更的列（{n}）",
        "type.no_changes": "没有列需要类型转换。",

        # ── 缺失值 Tab ──
        "missing.no_missing": "没有发现缺失值 🎉",

        # ── 异常值 Tab ──
        "outlier.caption": "基于 IQR 方法检测异常值（乘数 = {k}）。",
        "outlier.no_outliers": "数值列中未检测到异常值 🎉",

        # ── 脱敏 Tab ──
        "masking.header": "**🔒 已脱敏列**",
        "masking.total": "脱敏列总数：{n}",
        "masking.none": "未检测到或未脱敏任何敏感列。",
        "masking.detected_hint": "🔍 **已检测到敏感列：** ",

        # ── 列名变更 Tab ──
        "rename.already_standard": "所有列名已符合标准格式。",

        # ── 下载区域 ──
        "download.header": "⬇️ 下载清洗后的数据集",
        "download.csv": "📥 下载 CSV",
        "download.excel": "📥 下载 Excel",
        "download.report_expander": "📋 完整清洗报告（JSON）",

        # ── 原始数据预览 ──
        "original.preview": "📄 原始数据预览",
        "original.dtype_expander": "原始列类型和缺失值",

        # ── 通用 ──
        "common.before": "清洗前",
        "common.after": "清洗后",
        "common.column": "列名",
        "common.action": "操作",
        "common.original_type": "原始类型",
        "common.new_type": "新类型",
        "common.detail": "说明",
    },
    "en": {
        # ── Page framework ──
        "page.title": "🧹 Automated Data Cleaning Studio",
        "page.caption": "Upload a CSV or Excel file → automatic type detection, deduplication, "
                       "missing-value & outlier handling, data masking → download a clean dataset.",

        # ── Language switch ──
        "lang.zh": "中文",
        "lang.en": "English",
        "lang.label": "语言 / Language",

        # ── Intro when no file ──
        "intro.hint": "👆 Upload a CSV or Excel file to get started. The app will automatically detect column "
                     "types, clean text, remove duplicates, handle missing values, treat outliers, "
                     "and optionally mask sensitive data.",
        "intro.header": "What this app does",
        "intro.card1.title": "🔍 Smart Type Detection",
        "intro.card1.desc": "Columns stored as text (e.g. `\"$50,000\"`, `\"2021-01-15\"`, `\"Yes\"/\"No\"`) "
                            "are automatically converted to numeric, datetime, boolean, or category types.",
        "intro.card2.title": "🧼 Cleaning",
        "intro.card2.desc": "Column names are standardized to `snake_case`, whitespace is trimmed, "
                            "and exact duplicate rows are removed.",
        "intro.card3.title": "📊 Missing Values & Outliers",
        "intro.card3.desc": "Missing values are imputed based on column type (median/mode/interpolation). "
                            "Outliers are detected via the IQR method and capped or removed.",
        "intro.card4.title": "🔒 Data Masking",
        "intro.card4.desc": "Sensitive data (names, phones, emails, ID cards) can be auto-detected "
                            "and masked to protect privacy.",

        # ── Sidebar ──
        "sidebar.header": "⚙️ Cleaning Options",
        "sidebar.missing_values": "Missing values",
        "sidebar.numeric_cols": "Numeric columns",
        "sidebar.numeric_help": "How to fill missing numeric values.",
        "sidebar.numeric.median": "median",
        "sidebar.numeric.mean": "mean",
        "sidebar.numeric.zero": "zero",
        "sidebar.categorical_cols": "Categorical / text columns",
        "sidebar.categorical_help": "'mode' fills with the most frequent value; 'missing_label' fills with the literal text 'Missing'.",
        "sidebar.categorical.mode": "mode",
        "sidebar.categorical.missing_label": "missing_label",
        "sidebar.outliers": "Outliers (numeric columns)",
        "sidebar.outlier_method": "Outlier strategy",
        "sidebar.outlier.cap": "Cap to IQR bounds (winsorize)",
        "sidebar.outlier.remove": "Remove rows with outliers",
        "sidebar.outlier.none": "Detect only — don't modify",
        "sidebar.iqr_help": "Lower = more aggressive outlier detection. 1.5 is the standard statistical convention.",
        "sidebar.masking_header": "🔒 Data Masking",
        "sidebar.masking_enable": "Enable data masking",
        "sidebar.masking_enable_help": "Auto-detect and mask sensitive data (names, phones, emails, ID cards, addresses).",
        "sidebar.masking_caption": "Mask the following types of sensitive data:",
        "sidebar.masking.name": "Names",
        "sidebar.masking.phone": "Phone numbers",
        "sidebar.masking.email": "Emails",
        "sidebar.masking.id_card": "ID cards",
        "sidebar.masking.address": "Addresses",
        "sidebar.pipeline_order": (
            "**Pipeline order:**\n"
            "1. Standardize column names\n"
            "2. Trim whitespace\n"
            "3. Remove duplicate rows\n"
            "4. Auto-detect & convert types\n"
            "5. Handle missing values\n"
            "6. Handle outliers\n"
            "7. Mask sensitive data (if enabled)"
        ),

        # ── Progress ──
        "progress.status": "🧹 Cleaning your data...",
        "progress.complete": "✅ Cleaning complete!",
        "progress.detecting": "Detecting sensitive columns...",
        "progress.done": "Done!",
        "progress.step_fmt": "Step {step}/{total}: {name}...",
        "progress.step.0": "Standardize column names",
        "progress.step.1": "Trim whitespace",
        "progress.step.2": "Remove duplicates",
        "progress.step.3": "Detect & convert types",
        "progress.step.4": "Re-normalize text",
        "progress.step.5": "Handle missing values",
        "progress.step.6": "Handle outliers",
        "progress.step.7": "Mask sensitive data",

        # ── Summary metrics ──
        "metrics.original_rows": "Original Rows",
        "metrics.final_rows": "Final Rows",
        "metrics.duplicates_removed": "Duplicates Removed",
        "metrics.columns_retyped": "Columns Retyped",
        "metrics.quality_score": "Quality Score",

        # ── Tabs ──
        "tab.cleaned_preview": "🧾 Cleaned Preview",
        "tab.quality_report": "📋 Quality Report",
        "tab.visualizations": "📊 Visualizations",
        "tab.type_conversions": "🔄 Type Conversions",
        "tab.missing_values": "❓ Missing Values",
        "tab.outliers": "📈 Outliers",
        "tab.masking": "🔒 Masking",
        "tab.column_renames": "🏷️ Column Renames",

        # ── Cleaned Preview ──
        "preview.showing_rows": "Showing first 20 of {n} rows.",

        # ── Quality Report ──
        "quality.header": "📋 Data Quality Report",
        "quality.overall_score": "Overall Quality Score / 100",
        "quality.scoring": "**Scoring:**",
        "quality.scoring_detail": "Completeness (50%) + Uniqueness (20%) + Type Consistency (30%)",
        "quality.completeness_before": "Completeness (Before)",
        "quality.completeness_after": "Completeness (After)",
        "quality.dup_rate_before": "Duplicate Rate (Before)",
        "quality.dup_rate_after": "Duplicate Rate (After)",
        "quality.rows_removed": "Rows Removed",
        "quality.columns_retyped_label": "Columns Retyped",
        "quality.numeric_stats": "Numeric Column Statistics (After Cleaning)",

        # ── Visualizations ──
        "viz.missing_title": "Missing Values Comparison",
        "viz.missing_before": "Before Cleaning",
        "viz.missing_after": "After Cleaning",
        "viz.missing_xlabel": "Missing Count",
        "viz.missing_suptitle": "Missing Values: Before vs After",
        "viz.outlier_title": "Outlier Box Plots",
        "viz.outlier_suptitle": "Outlier Handling: Before vs After (Box Plots)",
        "viz.dist_title": "Distribution Histograms",
        "viz.dist_suptitle": "Distribution Changes: Before vs After",
        "viz.no_outlier_cols": "No outlier columns to visualize.",

        # ── Type Conversions ──
        "type.changed_header": "**✨ Columns automatically converted:**",
        "type.unchanged_expander": "Unchanged columns ({n})",
        "type.no_changes": "No columns needed type conversion.",

        # ── Missing Values ──
        "missing.no_missing": "No missing values found. 🎉",

        # ── Outliers ──
        "outlier.caption": "Outliers detected via IQR method (multiplier = {k}).",
        "outlier.no_outliers": "No outliers detected in numeric columns. 🎉",

        # ── Masking ──
        "masking.header": "**🔒 Masked Columns**",
        "masking.total": "Total columns masked: {n}",
        "masking.none": "No sensitive columns detected or masked.",
        "masking.detected_hint": "🔍 **Sensitive columns detected:** ",

        # ── Column Renames ──
        "rename.already_standard": "All column names were already standardized.",

        # ── Download ──
        "download.header": "⬇️ Download Cleaned Dataset",
        "download.csv": "📥 Download as CSV",
        "download.excel": "📥 Download as Excel",
        "download.report_expander": "📋 Full cleaning report (JSON)",

        # ── Original preview ──
        "original.preview": "📄 Original Data Preview",
        "original.dtype_expander": "Original column types & missing values",

        # ── Common ──
        "common.before": "Before",
        "common.after": "After",
        "common.column": "column",
        "common.action": "action",
        "common.original_type": "original_type",
        "common.new_type": "new_type",
        "common.detail": "detail",
    },
}

# ── Column display name mapping (CSV column → display name) ──────────────
COLUMN_DISPLAY_NAMES = {
    "zh": {
        "employee_name": "员工姓名",
        "salary": "薪资",
        "join_date": "入职日期",
        "is_active": "是否在职",
        "department": "部门",
        "age": "年龄",
        "performance_rating": "绩效评分",
        "phone": "手机号",
        "email": "邮箱",
        "id_card": "身份证号",
        "address": "地址",
        "name": "姓名",
        "score": "分数",
        "category": "分类",
        "active": "状态",
        "date": "日期",
        "value": "数值",
        "flag": "标记",
        "text": "文本",
        "num": "数值",
        "amount": "金额",
        "mixed": "混合内容",
        "dept": "部门",
        "mostly_empty": "大部分为空",
    },
    "en": {},
}


def t(key: str, **fmt) -> str:
    """Look up translated text for current language. Use `{key}` format strings."""
    lang = st.session_state.get("lang", "zh")
    text = T.get(lang, T["zh"]).get(key)
    if text is None:
        # Fallback: try the key itself for missing translations
        text = T["en"].get(key, key)
    if fmt:
        text = text.format(**fmt)
    return text


def translate_column_name(col: str) -> str:
    """Translate a column name for display, if a mapping exists."""
    lang = st.session_state.get("lang", "zh")
    mapping = COLUMN_DISPLAY_NAMES.get(lang, {})
    return mapping.get(col, col)


def translate_columns_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the DataFrame with translated column names for display."""
    lang = st.session_state.get("lang", "zh")
    if lang == "en":
        return df
    mapping = COLUMN_DISPLAY_NAMES.get("zh", {})
    rename_map = {c: mapping.get(c, c) for c in df.columns}
    return df.rename(columns=rename_map)


# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title=t("page.title") if "lang" in st.session_state else "🧹 自动化数据清洗平台",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize session state defaults ─────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "zh"  # 默认中文 / default Chinese

# ── Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 14px 18px; text-align: center;
}
.metric-card .num { font-size: 1.6rem; font-weight: 700; color: #1e293b; }
.metric-card .lbl { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
.section-header { font-size: 1.05rem; font-weight: 600; color: #1e293b; margin: 0.5rem 0; }
.quality-score {
    font-size: 3rem; font-weight: 800; text-align: center;
    background: linear-gradient(135deg, #22c55e, #3b82f6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.lang-container {
    display: flex; justify-content: flex-end; align-items: center;
    gap: 8px; padding: 4px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Top bar: title + language switcher ────────────────────────────────────
title_col, lang_col = st.columns([4, 1])
with title_col:
    st.title(t("page.title"))
    st.caption(t("page.caption"))
with lang_col:
    st.markdown("<br>", unsafe_allow_html=True)  # vertical alignment spacer
    current_lang_label = t("lang.zh") if st.session_state.lang == "zh" else t("lang.en")
    selected_lang = st.selectbox(
        t("lang.label"),
        options=["zh", "en"],
        format_func=lambda x: t("lang.zh") if x == "zh" else t("lang.en"),
        index=0 if st.session_state.lang == "zh" else 1,
        key="lang_selector",
        label_visibility="collapsed",
    )
    # Sync to session state
    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang
        st.rerun()

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar: cleaning options
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header(t("sidebar.header"))

    st.markdown(f"**{t('sidebar.missing_values')}**")
    numeric_strategy = st.selectbox(
        t("sidebar.numeric_cols"),
        ["median", "mean", "zero"],
        format_func=lambda x: {
            "median": t("sidebar.numeric.median"),
            "mean": t("sidebar.numeric.mean"),
            "zero": t("sidebar.numeric.zero"),
        }[x],
        help=t("sidebar.numeric_help"),
    )
    categorical_strategy = st.selectbox(
        t("sidebar.categorical_cols"),
        ["mode", "missing_label"],
        format_func=lambda x: {
            "mode": t("sidebar.categorical.mode"),
            "missing_label": t("sidebar.categorical.missing_label"),
        }[x],
        help=t("sidebar.categorical_help"),
    )

    st.markdown(f"**{t('sidebar.outliers')}**")
    outlier_method = st.selectbox(
        t("sidebar.outlier_method"),
        ["cap", "remove", "none"],
        format_func=lambda x: {
            "cap": t("sidebar.outlier.cap"),
            "remove": t("sidebar.outlier.remove"),
            "none": t("sidebar.outlier.none"),
        }[x],
    )
    iqr_multiplier = st.slider(
        "IQR multiplier (sensitivity)", 1.0, 3.0, 1.5, 0.1,
        help=t("sidebar.iqr_help"),
    )

    st.divider()

    # ── Data Masking ──────────────────────────────────────────────────────
    st.header(t("sidebar.masking_header"))
    enable_masking = st.checkbox(
        t("sidebar.masking_enable"),
        value=False,
        help=t("sidebar.masking_enable_help"),
    )

    masking_categories = {}
    detected_sensitive = {}

    if enable_masking:
        st.caption(t("sidebar.masking_caption"))
        masking_categories["mask_name"] = st.checkbox(t("sidebar.masking.name"), value=True)
        masking_categories["mask_phone"] = st.checkbox(t("sidebar.masking.phone"), value=True)
        masking_categories["mask_email"] = st.checkbox(t("sidebar.masking.email"), value=True)
        masking_categories["mask_id_card"] = st.checkbox(t("sidebar.masking.id_card"), value=True)
        masking_categories["mask_address"] = st.checkbox(t("sidebar.masking.address"), value=False)

    st.divider()
    st.caption(t("sidebar.pipeline_order"))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_file(file) -> pd.DataFrame:
    """Load CSV or Excel file into a DataFrame."""
    filename = file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(file)
    elif filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    else:
        st.error(f"Unsupported file format: {file.name}")
        st.stop()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned Data")
    return buf.getvalue()


def dtype_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        t("common.column"): df.columns,
        "dtype": [str(dt) for dt in df.dtypes],
        "missing": df.isna().sum().values,
        "missing_%": (df.isna().mean() * 100).round(1).values,
        "unique": [df[c].nunique() for c in df.columns],
    })
    return summary


def display_df_with_translated_cols(df: pd.DataFrame, **kwargs):
    """Display a DataFrame with column names translated for current language."""
    display_df = translate_columns_in_df(df)
    st.dataframe(display_df, **kwargs)


# ── Visualization helpers ─────────────────────────────────────────────────
def plot_missing_values(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame,
                         reverse_rename: dict | None = None):
    """Bar chart comparing missing values before/after cleaning."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Build reverse mapping: cleaned_name → original_name in raw_df
    if reverse_rename is None:
        reverse_rename = {}

    raw_missing = raw_df.isna().sum()
    cleaned_missing = cleaned_df.isna().sum()

    # Use cleaned_df column order; map back to raw column names for raw_df access
    cleaned_cols = cleaned_df.columns.tolist()
    raw_cols_for_display = [reverse_rename.get(c, c) for c in cleaned_cols]
    # Translate column names for the chart
    display_cols = [translate_column_name(c) for c in raw_cols_for_display]

    # Align raw_missing to cleaned column order
    raw_values = [raw_missing.get(reverse_rename.get(c, c), 0) for c in cleaned_cols]

    ax[0].barh(range(len(display_cols)), raw_values, color="#ef4444", alpha=0.7)
    ax[0].set_yticks(range(len(display_cols)))
    ax[0].set_yticklabels(display_cols, fontsize=9)
    ax[0].set_title(t("viz.missing_before"), fontsize=13, fontweight="bold")
    ax[0].set_xlabel(t("viz.missing_xlabel"))
    ax[0].invert_yaxis()

    ax[1].barh(range(len(display_cols)), cleaned_missing.values, color="#22c55e", alpha=0.7)
    ax[1].set_yticks(range(len(display_cols)))
    ax[1].set_yticklabels(display_cols, fontsize=9)
    ax[1].set_title(t("viz.missing_after"), fontsize=13, fontweight="bold")
    ax[1].set_xlabel(t("viz.missing_xlabel"))
    ax[1].invert_yaxis()

    fig.suptitle(t("viz.missing_suptitle"), fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_outlier_boxplots(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame,
                          numeric_cols: list[str], outlier_report: dict,
                          reverse_rename: dict | None = None):
    """Side-by-side boxplots for numeric columns with outliers."""
    outlier_cols = [c for c in numeric_cols if c in outlier_report]
    if not outlier_cols:
        return None

    if reverse_rename is None:
        reverse_rename = {}

    n_cols = min(len(outlier_cols), 4)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(outlier_cols[:4]):
        ax = axes[i]
        # Map cleaned column name back to original column name in raw_df
        raw_col = reverse_rename.get(col, col)
        data_before = raw_df[raw_col].dropna()
        data_after = cleaned_df[col].dropna()

        # raw_df data may still be strings (e.g. "$50,000"); coerc to numeric
        data_before = pd.to_numeric(data_before, errors="coerce").dropna()
        data_after = pd.to_numeric(data_after, errors="coerce").dropna()

        bp = ax.boxplot(
            [data_before, data_after],
            patch_artist=True,
            widths=0.5,
        )
        ax.set_xticklabels([t("common.before"), t("common.after")])
        bp["boxes"][0].set_facecolor("#fca5a5")
        bp["boxes"][1].set_facecolor("#86efac")
        display_name = translate_column_name(col)
        ax.set_title(display_name, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)

    fig.suptitle(t("viz.outlier_suptitle"), fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_distribution_histograms(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame,
                                 numeric_cols: list[str],
                                 reverse_rename: dict | None = None):
    """Overlaid histograms showing distribution changes for numeric columns."""
    if not numeric_cols:
        return None

    if reverse_rename is None:
        reverse_rename = {}

    n_cols = min(len(numeric_cols), 3)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(numeric_cols[:3]):
        ax = axes[i]
        raw_col = reverse_rename.get(col, col)
        # raw data may still be strings; convert to numeric
        raw_data = pd.to_numeric(raw_df[raw_col], errors="coerce").dropna()
        clean_data = pd.to_numeric(cleaned_df[col], errors="coerce").dropna()
        ax.hist(raw_data, bins=25, alpha=0.5, label=t("common.before"),
                color="#ef4444", edgecolor="white")
        ax.hist(clean_data, bins=25, alpha=0.5, label=t("common.after"),
                color="#3b82f6", edgecolor="white")
        display_name = translate_column_name(col)
        ax.set_title(display_name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    fig.suptitle(t("viz.dist_suptitle"), fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def compute_quality_score(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame,
                          report: dict) -> dict:
    """Compute data quality metrics."""
    raw_shape = raw_df.shape
    cleaned_shape = cleaned_df.shape

    raw_completeness = 1 - (raw_df.isna().sum().sum() / max(raw_shape[0] * raw_shape[1], 1))
    cleaned_completeness = 1 - (cleaned_df.isna().sum().sum() / max(cleaned_shape[0] * cleaned_shape[1], 1))

    raw_dup_rate = raw_df.duplicated().sum() / max(raw_shape[0], 1)
    cleaned_dup_rate = cleaned_df.duplicated().sum() / max(cleaned_shape[0], 1)

    n_converted = sum(
        1 for v in report.get("type_conversions", {}).values()
        if "already" not in v.get("note", "") and "free text" not in v.get("note", "")
    )
    total_cols = max(len(raw_df.columns), 1)

    score = (
        cleaned_completeness * 50 +
        (1 - cleaned_dup_rate) * 20 +
        (n_converted / total_cols) * 30
    )

    return {
        "quality_score": round(score, 1),
        "raw_completeness": round(raw_completeness * 100, 1),
        "cleaned_completeness": round(cleaned_completeness * 100, 1),
        "raw_dup_rate": round(raw_dup_rate * 100, 2),
        "cleaned_dup_rate": round(cleaned_dup_rate * 100, 2),
        "columns_retyped": n_converted,
        "total_columns": total_cols,
        "rows_removed": raw_shape[0] - cleaned_shape[0],
        "duplicates_removed": report.get("duplicates_removed", 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# File upload
# ══════════════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file" if st.session_state.lang == "en" else "上传 CSV 或 Excel 文件",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is None:
    st.info(t("intro.hint"))
    st.markdown(f"### {t('intro.header')}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"**{t('intro.card1.title')}**")
        st.write(t("intro.card1.desc"))
    with c2:
        st.markdown(f"**{t('intro.card2.title')}**")
        st.write(t("intro.card2.desc"))
    with c3:
        st.markdown(f"**{t('intro.card3.title')}**")
        st.write(t("intro.card3.desc"))
    with c4:
        st.markdown(f"**{t('intro.card4.title')}**")
        st.write(t("intro.card4.desc"))

else:
    raw_df = load_file(uploaded_file)

    st.markdown(f'<div class="section-header">{t("original.preview")}</div>', unsafe_allow_html=True)
    display_df_with_translated_cols(raw_df.head(10), use_container_width=True)

    with st.expander(t("original.dtype_expander")):
        st.dataframe(dtype_summary(raw_df), use_container_width=True)

    st.divider()

    # ── Auto-detect sensitive columns (if masking is enabled) ──
    masking_columns = None
    if enable_masking:
        with st.spinner(t("progress.detecting")):
            masking_columns = detect_sensitive_columns(raw_df)
        if masking_columns:
            st.info(
                t("masking.detected_hint") +
                ", ".join(
                    f"{cat}: {', '.join(cols)}"
                    for cat, cols in masking_columns.items()
                )
            )

    # ── Build translated progress step names ──────────────────────────
    progress_step_names = {
        i: t(f"progress.step.{i}")
        for i in range(8)
    }

    # ── Run cleaning with progress ────────────────────────────────────
    with st.status(t("progress.status"), expanded=True) as status:
        progress_placeholder = st.empty()

        def progress_callback(idx, total, name):
            # Use translated step name
            translated_name = progress_step_names.get(idx, name)
            progress_placeholder.text(
                t("progress.step_fmt", step=idx + 1, total=total, name=translated_name)
            )

        cleaned_df, report = clean_dataframe(
            raw_df,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            outlier_method=outlier_method,
            iqr_multiplier=iqr_multiplier,
            progress_callback=progress_callback,
            enable_masking=enable_masking,
            masking_columns=masking_columns,
            masking_categories=masking_categories if enable_masking else None,
        )
        progress_placeholder.text(t("progress.done"))
        status.update(label=t("progress.complete"), state="complete", expanded=False)

    # ── Compute quality metrics ──
    quality = compute_quality_score(raw_df, cleaned_df, report)

    # ── Summary metrics ──
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="num">{report["original_shape"][0]:,}</div>'
                    f'<div class="lbl">{t("metrics.original_rows")}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="num">{report["final_shape"][0]:,}</div>'
                    f'<div class="lbl">{t("metrics.final_rows")}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="num">{quality["duplicates_removed"]:,}</div>'
                    f'<div class="lbl">{t("metrics.duplicates_removed")}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="num">{quality["columns_retyped"]}</div>'
                    f'<div class="lbl">{t("metrics.columns_retyped")}</div></div>', unsafe_allow_html=True)
    with m5:
        score = quality["quality_score"]
        st.markdown(f'<div class="metric-card"><div class="num">{score:.0f}/100</div>'
                    f'<div class="lbl">{t("metrics.quality_score")}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab_names = [
        t("tab.cleaned_preview"),
        t("tab.quality_report"),
        t("tab.visualizations"),
        t("tab.type_conversions"),
        t("tab.missing_values"),
        t("tab.outliers"),
    ]
    if enable_masking:
        tab_names.append(t("tab.masking"))
    tab_names.append(t("tab.column_renames"))

    tabs = st.tabs(tab_names)
    tab_idx = 0

    # ── Tab: Cleaned Preview ──────────────────────────────────────────
    with tabs[tab_idx]:
        tab_idx += 1
        display_df_with_translated_cols(cleaned_df.head(20), use_container_width=True)
        st.caption(t("preview.showing_rows", n=f"{len(cleaned_df):,}"))
        st.dataframe(dtype_summary(cleaned_df), use_container_width=True)

    # ── Tab: Quality Report ───────────────────────────────────────────
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown(f'<div class="section-header">{t("quality.header")}</div>', unsafe_allow_html=True)

        score_col, stats_col = st.columns([1, 2])

        with score_col:
            st.markdown(f'<div class="quality-score">{quality["quality_score"]:.0f}</div>', unsafe_allow_html=True)
            st.caption(t("quality.overall_score"))
            st.markdown(t("quality.scoring"))
            st.caption(t("quality.scoring_detail"))

        with stats_col:
            qm1, qm2, qm3 = st.columns(3)
            with qm1:
                st.metric(
                    t("quality.completeness_before"),
                    f"{quality['raw_completeness']}%",
                    delta=f"{quality['cleaned_completeness'] - quality['raw_completeness']:+.1f}%",
                )
                st.metric(t("quality.completeness_after"), f"{quality['cleaned_completeness']}%")
            with qm2:
                st.metric(
                    t("quality.dup_rate_before"),
                    f"{quality['raw_dup_rate']}%",
                    delta=f"{quality['cleaned_dup_rate'] - quality['raw_dup_rate']:+.2f}%",
                )
                st.metric(t("quality.dup_rate_after"), f"{quality['cleaned_dup_rate']}%")
            with qm3:
                st.metric(t("quality.rows_removed"), f"{quality['rows_removed']:,}")
                st.metric(
                    t("quality.columns_retyped_label"),
                    f"{quality['columns_retyped']}/{quality['total_columns']}",
                )

        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.markdown(f"**{t('quality.numeric_stats')}**")
            st.dataframe(
                cleaned_df[numeric_cols].describe().T.style.format("{:.2f}"),
                use_container_width=True,
            )

    # ── Tab: Visualizations ───────────────────────────────────────────
    with tabs[tab_idx]:
        tab_idx += 1
        st.markdown(f'<div class="section-header">{t("tab.visualizations")}</div>', unsafe_allow_html=True)

        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()

        # Build reverse column name mapping: cleaned_name → original_name
        # Needed so plotting functions can look up columns in raw_df (which
        # still has the original column names from the uploaded file).
        reverse_rename = {v: k for k, v in report.get("column_renames", {}).items()}

        # 1. Missing values comparison
        st.markdown(f"**{t('viz.missing_title')}**")
        fig_missing = plot_missing_values(raw_df, cleaned_df, reverse_rename)
        st.pyplot(fig_missing)
        plt.close(fig_missing)

        # 2. Outlier boxplots
        if numeric_cols and report.get("outliers"):
            st.markdown(f"**{t('viz.outlier_title')}**")
            fig_box = plot_outlier_boxplots(raw_df, cleaned_df, numeric_cols, report["outliers"], reverse_rename)
            if fig_box:
                st.pyplot(fig_box)
                plt.close(fig_box)
            else:
                st.caption(t("viz.no_outlier_cols"))

        # 3. Distribution histograms
        if numeric_cols:
            st.markdown(f"**{t('viz.dist_title')}**")
            fig_dist = plot_distribution_histograms(raw_df, cleaned_df, numeric_cols, reverse_rename)
            if fig_dist:
                st.pyplot(fig_dist)
                plt.close(fig_dist)

    # ── Tab: Type Conversions ─────────────────────────────────────────
    with tabs[tab_idx]:
        tab_idx += 1
        if report["type_conversions"]:
            rows = []
            for col, info in report["type_conversions"].items():
                rows.append({
                    t("common.column"): col,
                    t("common.original_type"): info["original"],
                    t("common.new_type"): info["converted_to"],
                    t("common.detail"): info["note"],
                })
            conv_df = pd.DataFrame(rows)
            changed = conv_df[~conv_df[t("common.detail")].str.contains("already|free text", regex=True)]
            unchanged = conv_df[conv_df[t("common.detail")].str.contains("already|free text", regex=True)]
            if len(changed):
                st.markdown(t("type.changed_header"))
                st.dataframe(changed, use_container_width=True, hide_index=True)
            if len(unchanged):
                with st.expander(t("type.unchanged_expander", n=len(unchanged))):
                    st.dataframe(unchanged, use_container_width=True, hide_index=True)
        else:
            st.write(t("type.no_changes"))

    # ── Tab: Missing Values ───────────────────────────────────────────
    with tabs[tab_idx]:
        tab_idx += 1
        if report["missing_values"]:
            rows = [{"column": c, **v} for c, v in report["missing_values"].items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.write(t("missing.no_missing"))

    # ── Tab: Outliers ─────────────────────────────────────────────────
    with tabs[tab_idx]:
        tab_idx += 1
        if report["outliers"]:
            rows = []
            for c, v in report["outliers"].items():
                rows.append({
                    "column": c,
                    "n_outliers": v["n_outliers"],
                    "lower_bound": v["bounds"][0],
                    "upper_bound": v["bounds"][1],
                    "action": v["action"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(t("outlier.caption", k=f"{iqr_multiplier}"))
        else:
            st.write(t("outlier.no_outliers"))

    # ── Tab: Masking (only if enabled) ────────────────────────────────
    if enable_masking:
        with tabs[tab_idx]:
            tab_idx += 1
            if report.get("masking") and report["masking"]["masked_columns"]:
                st.markdown(t("masking.header"))
                st.dataframe(
                    pd.DataFrame(report["masking"]["masked_columns"]),
                    use_container_width=True, hide_index=True,
                )
                st.caption(t("masking.total", n=report["masking"]["total_masked"]))
            else:
                st.write(t("masking.none"))

    # ── Tab: Column Renames ───────────────────────────────────────────
    with tabs[tab_idx]:
        tab_idx += 1
        if report["column_renames"]:
            rows = [{"original_name": k, "standardized_name": v}
                   for k, v in report["column_renames"].items() if k != v]
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.write(t("rename.already_standard"))

    st.divider()

    # ── Download ──────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">{t("download.header")}</div>', unsafe_allow_html=True)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        csv_bytes = df_to_csv_bytes(cleaned_df)
        st.download_button(
            label=t("download.csv"),
            data=csv_bytes,
            file_name=f"cleaned_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )
    with dl_col2:
        excel_bytes = df_to_excel_bytes(cleaned_df)
        st.download_button(
            label=t("download.excel"),
            data=excel_bytes,
            file_name=f"cleaned_{uploaded_file.name.rsplit('.', 1)[0]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with st.expander(t("download.report_expander")):
        def _default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (pd.Timestamp,)):
                return str(o)
            return str(o)
        st.json(json.loads(json.dumps(report, default=_default)))
