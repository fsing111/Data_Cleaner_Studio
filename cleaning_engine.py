"""
Automated Data Cleaning Engine
================================
Pure-logic module: detects column types, cleans, standardizes,
handles missing values and outliers. No Streamlit dependencies —
fully unit-testable on its own.
"""

import re
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────
# 1. COLUMN NAME STANDARDIZATION
# ──────────────────────────────────────────────────────────────────────────
def standardize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Convert column names to snake_case, strip whitespace,
    remove special characters, and resolve duplicates.
    Returns (df, rename_map).
    """
    rename_map = {}
    seen = {}

    for col in df.columns:
        new_col = str(col).strip()
        new_col = re.sub(r"[^\w\s]", "", new_col)        # drop special chars
        new_col = re.sub(r"\s+", "_", new_col)             # spaces -> underscore
        new_col = re.sub(r"_+", "_", new_col)              # collapse repeats
        new_col = new_col.strip("_").lower()
        if new_col == "":
            new_col = "unnamed_column"

        # de-duplicate
        if new_col in seen:
            seen[new_col] += 1
            new_col = f"{new_col}_{seen[new_col]}"
        else:
            seen[new_col] = 0

        rename_map[col] = new_col

    df = df.rename(columns=rename_map)
    return df, rename_map


# ──────────────────────────────────────────────────────────────────────────
# 2. DUPLICATE REMOVAL
# ──────────────────────────────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    return df, removed


# ──────────────────────────────────────────────────────────────────────────
# 3. SMART TYPE DETECTION
# ──────────────────────────────────────────────────────────────────────────
NUMERIC_CLEAN_RE = re.compile(r"[,\$\€\£\%\s]")

# Common date formats to try (helps disambiguate ambiguous strings)
DATE_HINT_WORDS = {"date", "time", "dob", "day", "month", "year", "timestamp", "created", "updated"}

BOOLEAN_TRUE_SET  = {"true", "yes", "y", "1", "t"}
BOOLEAN_FALSE_SET = {"false", "no", "n", "0", "f"}


def _try_numeric_conversion(series: pd.Series) -> pd.Series | None:
    """Attempt to convert an object series to numeric by stripping
    common currency/percent/thousands formatting. Returns converted
    series if >= 80% of non-null values parse successfully, else None."""
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(NUMERIC_CLEAN_RE, "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (123) -> -123 (accounting negatives)
    )
    converted = pd.to_numeric(cleaned, errors="coerce")

    non_null_mask = series.notna()
    if non_null_mask.sum() == 0:
        return None

    success_rate = converted[non_null_mask].notna().mean()
    if success_rate >= 0.8:
        return converted
    return None


def _try_datetime_conversion(series: pd.Series, col_name: str = "") -> pd.Series | None:
    """Attempt to convert an object series to datetime.
    Returns converted series if >= 80% of non-null values parse, else None."""
    non_null_mask = series.notna()
    if non_null_mask.sum() == 0:
        return None

    sample = series[non_null_mask].astype(str).str.strip()

    # Quick reject: must look date-ish (contains digits and separators or month names)
    date_pattern = re.compile(
        r"\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}|"
        r"\d{4}\d{2}\d{2}|"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        re.IGNORECASE,
    )
    looks_date = sample.str.contains(date_pattern, regex=True, na=False)
    if looks_date.mean() < 0.5:
        return None

    converted = pd.to_datetime(series, errors="coerce", format="mixed")
    success_rate = converted[non_null_mask].notna().mean()

    if success_rate >= 0.8:
        return converted
    return None


def _try_boolean_conversion(series: pd.Series) -> pd.Series | None:
    """Detect boolean-like columns (Yes/No, True/False, 1/0, Y/N)."""
    non_null_mask = series.notna()
    if non_null_mask.sum() == 0:
        return None

    unique_vals = set(series[non_null_mask].astype(str).str.strip().str.lower().unique())
    if not unique_vals:
        return None

    if unique_vals.issubset(BOOLEAN_TRUE_SET | BOOLEAN_FALSE_SET) and len(unique_vals) <= 2:
        mapped = series.astype(str).str.strip().str.lower().map(
            lambda x: True if x in BOOLEAN_TRUE_SET else
                      (False if x in BOOLEAN_FALSE_SET else np.nan)
        )
        return mapped
    return None


def _try_categorical_conversion(series: pd.Series, max_unique_ratio: float = 0.5,
                                 max_unique_abs: int = 50) -> bool:
    """Decide if a low-cardinality object column should become 'category' dtype."""
    n = len(series)
    if n == 0:
        return False
    nunique = series.nunique(dropna=True)
    if nunique == 0:
        return False
    ratio = nunique / n
    return nunique <= max_unique_abs and ratio <= max_unique_ratio


def detect_and_convert_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    For every object/string column, attempt — in priority order — to
    convert to: numeric -> datetime -> boolean -> category -> leave as text.

    Returns (df, type_report) where type_report maps
    column -> {'original': str, 'converted_to': str, 'success_rate': float}
    """
    df = df.copy()
    type_report = {}

    for col in df.columns:
        original_dtype = str(df[col].dtype)

        # Skip columns that are already well-typed numeric/datetime/bool
        if pd.api.types.is_numeric_dtype(df[col]):
            type_report[col] = {"original": original_dtype, "converted_to": original_dtype, "note": "already numeric"}
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            type_report[col] = {"original": original_dtype, "converted_to": original_dtype, "note": "already datetime"}
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            type_report[col] = {"original": original_dtype, "converted_to": original_dtype, "note": "already boolean"}
            continue

        # Only attempt conversion on object/string columns
        series = df[col]
        converted = False

        # 1. Try numeric (e.g. "salary" stored as "$50,000")
        num_result = _try_numeric_conversion(series)
        if num_result is not None:
            df[col] = num_result
            type_report[col] = {"original": original_dtype, "converted_to": str(num_result.dtype),
                                 "note": "numeric (cleaned currency/percent/commas)"}
            converted = True

        # 2. Try datetime (e.g. mixed date strings)
        if not converted:
            dt_result = _try_datetime_conversion(series, col_name=col)
            if dt_result is not None:
                df[col] = dt_result
                type_report[col] = {"original": original_dtype, "converted_to": "datetime64[ns]",
                                     "note": "parsed as datetime"}
                converted = True

        # 3. Try boolean (Yes/No, True/False, Y/N)
        if not converted:
            bool_result = _try_boolean_conversion(series)
            if bool_result is not None:
                df[col] = bool_result.astype("boolean")
                type_report[col] = {"original": original_dtype, "converted_to": "boolean",
                                     "note": "Yes/No or True/False detected"}
                converted = True

        # 4. Try categorical (low-cardinality text)
        if not converted:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
            if _try_categorical_conversion(df[col]):
                df[col] = df[col].astype("category")
                type_report[col] = {"original": original_dtype, "converted_to": "category",
                                     "note": f"low-cardinality text ({df[col].nunique()} unique)"}
            else:
                type_report[col] = {"original": original_dtype, "converted_to": "object (text, trimmed)",
                                     "note": "free text — kept as string"}

    return df, type_report


# ──────────────────────────────────────────────────────────────────────────
# 4. MISSING VALUE HANDLING
# ──────────────────────────────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame,
                          numeric_strategy: str = "median",
                          categorical_strategy: str = "mode",
                          datetime_strategy: str = "interpolate",
                          drop_thresh: float = 0.6) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing values based on column dtype.

    numeric_strategy:     'median' | 'mean' | 'zero'
    categorical_strategy: 'mode'   | 'missing_label'
    datetime_strategy:    'interpolate' | 'drop_rows'
    drop_thresh:          drop a column entirely if missing fraction exceeds this
    """
    df = df.copy()
    report = {}

    cols_to_drop = []
    for col in df.columns:
        miss_frac = df[col].isna().mean()
        if miss_frac == 0:
            continue

        if miss_frac > drop_thresh:
            cols_to_drop.append(col)
            report[col] = {"missing_pct": round(miss_frac * 100, 1), "action": "dropped column (>60% missing)"}
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if numeric_strategy == "median":
                fill_val = df[col].median()
            elif numeric_strategy == "mean":
                fill_val = df[col].mean()
            else:
                fill_val = 0
            df[col] = df[col].fillna(fill_val)
            report[col] = {"missing_pct": round(miss_frac * 100, 1),
                           "action": f"filled with {numeric_strategy} ({fill_val:.2f})"}

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            if datetime_strategy == "interpolate":
                df[col] = df[col].interpolate(method="linear")
                # fallback: forward/back fill any remaining edges
                df[col] = df[col].ffill().bfill()
            report[col] = {"missing_pct": round(miss_frac * 100, 1), "action": "interpolated/forward-filled"}

        elif pd.api.types.is_bool_dtype(df[col]):
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if len(mode_val) else False
            df[col] = df[col].fillna(fill_val)
            report[col] = {"missing_pct": round(miss_frac * 100, 1), "action": f"filled with mode ({fill_val})"}

        else:  # category / object
            if categorical_strategy == "mode":
                mode_val = df[col].mode(dropna=True)
                fill_val = mode_val.iloc[0] if len(mode_val) else "Unknown"
            else:
                fill_val = "Missing"

            if isinstance(df[col].dtype, pd.CategoricalDtype) and fill_val not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories([fill_val])
            df[col] = df[col].fillna(fill_val)
            report[col] = {"missing_pct": round(miss_frac * 100, 1), "action": f"filled with '{fill_val}'"}

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df, report


# ──────────────────────────────────────────────────────────────────────────
# 5. OUTLIER HANDLING
# ──────────────────────────────────────────────────────────────────────────
def handle_outliers(df: pd.DataFrame, method: str = "cap", iqr_multiplier: float = 1.5,
                    exclude_cols: list | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Detect and treat outliers in numeric columns using the IQR method.

    method: 'cap'   -> clip values to [Q1 - k*IQR, Q3 + k*IQR]  (winsorize)
            'remove'-> drop rows containing any outlier (use cautiously)
            'none'  -> detect only, no modification
    exclude_cols: columns to skip (e.g. ID columns, binary flags)
    """
    df = df.copy()
    report = {}
    exclude_cols = set(exclude_cols or [])

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude_cols]

    rows_to_drop = set()

    for col in numeric_cols:
        # Skip binary/flag-like columns (only 0/1 or 2 unique values)
        if df[col].nunique(dropna=True) <= 2:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        outlier_mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = int(outlier_mask.sum())

        if n_outliers == 0:
            continue

        if method == "cap":
            df[col] = df[col].clip(lower=lower, upper=upper)
            action = f"capped {n_outliers} values to [{lower:.2f}, {upper:.2f}]"
        elif method == "remove":
            rows_to_drop.update(df.index[outlier_mask].tolist())
            action = f"flagged {n_outliers} rows for removal"
        else:
            action = f"detected {n_outliers} outliers (no action taken)"

        report[col] = {
            "n_outliers": n_outliers,
            "bounds": (round(float(lower), 2), round(float(upper), 2)),
            "action": action,
        }

    if method == "remove" and rows_to_drop:
        df = df.drop(index=list(rows_to_drop)).reset_index(drop=True)

    return df, report


# ──────────────────────────────────────────────────────────────────────────
# 6. WHITESPACE / STRING NORMALIZATION
# ──────────────────────────────────────────────────────────────────────────
def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace and collapse internal multi-spaces
    in object/category text columns."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        )
    return df


# ──────────────────────────────────────────────────────────────────────────
# 7. DATA MASKING / DESENSITIZATION
# ──────────────────────────────────────────────────────────────────────────
# Patterns for auto-detecting sensitive columns by content
_PHONE_PATTERN = re.compile(r"1[3-9]\d{9}")
_ID_CARD_PATTERN = re.compile(r"\d{17}[\dXx]")
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Column name keywords that suggest sensitive data (underscore/hyphen-free forms)
_SENSITIVE_KEYWORDS_NAME = {
    "name", "姓名", "名字", "full name", "first name", "last name", "username",
}
_SENSITIVE_KEYWORDS_PHONE = {
    "phone", "mobile", "tel", "电话", "手机", "cellphone", "phone number",
}
_SENSITIVE_KEYWORDS_EMAIL = {
    "email", "mail", "e-mail", "e mail", "邮箱", "电子邮箱",
}
_SENSITIVE_KEYWORDS_ID = {
    "id card", "id number", "身份证", "ssn", "passport", "护照",
    "id_card", "id_number",
}
_SENSITIVE_KEYWORDS_ADDRESS = {
    "address", "addr", "地址", "住址", "location",
}


def detect_sensitive_columns(df: pd.DataFrame, sample_size: int = 100) -> dict[str, list[str]]:
    """
    Auto-detect columns that may contain sensitive information.

    Detection methods:
    1. Column name keyword matching
    2. Content pattern matching (phone numbers, ID cards, emails)

    Returns a dict mapping category -> list of column names.
    Example: {'phone': ['mobile', 'contact'], 'name': ['employee_name'], ...}

    Categories: 'name', 'phone', 'id_card', 'email', 'address'
    """
    detected: dict[str, set[str]] = {
        "name": set(),
        "phone": set(),
        "id_card": set(),
        "email": set(),
        "address": set(),
    }

    for col in df.columns:
        col_lower = str(col).lower().replace("_", " ").replace("-", " ")

        # Method 1: keyword matching on column name
        # Also check the original column name (lowercased) to catch "id_card" etc.
        col_lower_orig = str(col).lower()
        if any(kw in col_lower for kw in _SENSITIVE_KEYWORDS_NAME) or \
           any(kw in col_lower_orig for kw in _SENSITIVE_KEYWORDS_NAME):
            detected["name"].add(col)
        if any(kw in col_lower for kw in _SENSITIVE_KEYWORDS_PHONE) or \
           any(kw in col_lower_orig for kw in _SENSITIVE_KEYWORDS_PHONE):
            detected["phone"].add(col)
        if any(kw in col_lower for kw in _SENSITIVE_KEYWORDS_EMAIL) or \
           any(kw in col_lower_orig for kw in _SENSITIVE_KEYWORDS_EMAIL):
            detected["email"].add(col)
        if any(kw in col_lower for kw in _SENSITIVE_KEYWORDS_ID) or \
           any(kw in col_lower_orig for kw in _SENSITIVE_KEYWORDS_ID):
            detected["id_card"].add(col)
        if any(kw in col_lower for kw in _SENSITIVE_KEYWORDS_ADDRESS) or \
           any(kw in col_lower_orig for kw in _SENSITIVE_KEYWORDS_ADDRESS):
            detected["address"].add(col)

        # Method 2: content pattern matching for string columns
        is_string_col = (
            df[col].dtype == object
            or isinstance(df[col].dtype, pd.StringDtype)
            or pd.api.types.is_string_dtype(df[col])
        )
        if is_string_col and col not in set().union(*detected.values()):
            sample = df[col].dropna().astype(str).head(sample_size)
            if len(sample) == 0:
                continue

            phone_hits = sample.str.match(_PHONE_PATTERN).mean()
            email_hits = sample.str.match(_EMAIL_PATTERN).mean()
            id_hits = sample.str.match(_ID_CARD_PATTERN).mean()

            if phone_hits >= 0.5:
                detected["phone"].add(col)
            if email_hits >= 0.5:
                detected["email"].add(col)
            if id_hits >= 0.5:
                detected["id_card"].add(col)

    # Convert sets to sorted lists, drop empty categories
    return {k: sorted(v) for k, v in detected.items() if v}


def mask_sensitive_data(
    df: pd.DataFrame,
    columns: dict[str, list[str]] | None = None,
    auto_detect: bool = True,
    mask_name: bool = True,
    mask_phone: bool = True,
    mask_id_card: bool = True,
    mask_email: bool = True,
    mask_address: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply masking to sensitive columns.

    Parameters
    ----------
    df : DataFrame to mask
    columns : optional pre-computed dict of {category: [col_names]}.
              If None and auto_detect=True, calls detect_sensitive_columns().
    auto_detect : if True and columns is None, auto-detect sensitive columns
    mask_name, mask_phone, mask_id_card, mask_email, mask_address :
        toggle which categories to mask

    Returns (masked_df, masking_report)
    """
    df = df.copy()

    if columns is None and auto_detect:
        columns = detect_sensitive_columns(df)

    if columns is None:
        return df, {"masked_columns": [], "note": "No columns to mask"}

    masked_cols: list[dict] = []

    for col in set().union(*columns.values()):
        if col not in df.columns:
            continue
        is_string_col = (
            df[col].dtype == object
            or isinstance(df[col].dtype, pd.StringDtype)
            or pd.api.types.is_string_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
        )
        if not is_string_col:
            continue

        # Determine category for this column
        col_categories = [cat for cat, cols in columns.items() if col in cols]

        series = df[col].astype(str)
        original_dtype = df[col].dtype

        for cat in col_categories:
            if cat == "name" and mask_name:
                # 张三 -> 张*, Alice -> A***
                def _mask_name(val: str) -> str:
                    if pd.isna(val) or val in ("nan", "None", ""):
                        return val
                    val = val.strip()
                    if len(val) <= 1:
                        return val
                    # Check if the string looks Chinese (first char is CJK)
                    if "一" <= val[0] <= "鿿" or "㐀" <= val[0] <= "䶿":
                        return val[0] + "*" * (len(val) - 1)
                    else:
                        return val[0] + "*" * min(len(val) - 1, 4)

                series = series.apply(_mask_name)
                masked_cols.append({"column": col, "category": cat, "method": "name_masking"})

            elif cat == "phone" and mask_phone:
                # 13812345678 -> 138****5678
                def _mask_phone(val: str) -> str:
                    if pd.isna(val) or val in ("nan", "None", ""):
                        return val
                    digits = re.sub(r"\D", "", str(val))
                    if len(digits) >= 7:
                        return digits[:3] + "****" + digits[-4:]
                    return re.sub(r"\d", "*", str(val))

                series = series.apply(_mask_phone)
                masked_cols.append({"column": col, "category": cat, "method": "phone_masking"})

            elif cat == "id_card" and mask_id_card:
                # 110101199001011234 -> 110101********1234
                def _mask_id_card(val: str) -> str:
                    if pd.isna(val) or val in ("nan", "None", ""):
                        return val
                    digits = re.sub(r"\D", "", str(val))
                    if len(digits) >= 10:
                        return digits[:6] + "********" + digits[-4:]
                    return re.sub(r"\d", "*", str(val))

                series = series.apply(_mask_id_card)
                masked_cols.append({"column": col, "category": cat, "method": "id_card_masking"})

            elif cat == "email" and mask_email:
                # johndoe@email.com -> j****@email.com
                def _mask_email(val: str) -> str:
                    if pd.isna(val) or val in ("nan", "None", ""):
                        return val
                    val = str(val).strip()
                    if "@" in val:
                        local, domain = val.split("@", 1)
                        if len(local) <= 1:
                            return "*@" + domain
                        return local[0] + "****@" + domain
                    return val

                series = series.apply(_mask_email)
                masked_cols.append({"column": col, "category": cat, "method": "email_masking"})

            elif cat == "address" and mask_address:
                # Keep first 3 chars, mask the rest
                def _mask_address(val: str) -> str:
                    if pd.isna(val) or val in ("nan", "None", ""):
                        return val
                    val = str(val).strip()
                    if len(val) <= 3:
                        return val
                    return val[:3] + "*" * min(len(val) - 3, 10)

                series = series.apply(_mask_address)
                masked_cols.append({"column": col, "category": cat, "method": "address_masking"})

        # Restore category dtype if original was category
        if isinstance(original_dtype, pd.CategoricalDtype):
            df[col] = series.astype("category")
        else:
            df[col] = series

    report = {
        "masked_columns": masked_cols,
        "total_masked": len(masked_cols),
    }
    return df, report


# ──────────────────────────────────────────────────────────────────────────
# 8. FULL PIPELINE
# ──────────────────────────────────────────────────────────────────────────
def clean_dataframe(df: pd.DataFrame,
                    numeric_strategy: str = "median",
                    categorical_strategy: str = "mode",
                    outlier_method: str = "cap",
                    iqr_multiplier: float = 1.5,
                    progress_callback=None,
                    enable_masking: bool = False,
                    masking_columns: dict | None = None,
                    masking_categories: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Run the full automated cleaning pipeline and return the cleaned
    DataFrame plus a structured report of every action taken.

    Parameters
    ----------
    progress_callback : callable(step_index, total_steps, step_name) or None
        If provided, called at each pipeline step for progress reporting.
    enable_masking : bool
        If True, run data masking after outlier handling.
    masking_columns : dict or None
        Pre-computed sensitive column mapping. Auto-detected if None.
    masking_categories : dict or None
        Toggle which categories to mask, e.g. {'mask_name': True, 'mask_phone': False}.
    """
    report = {}
    total_steps = 8 if enable_masking else 7

    def _step(idx: int, name: str):
        if progress_callback:
            progress_callback(idx, total_steps, name)

    original_shape = df.shape

    # 1. Standardize column names
    _step(0, "标准化列名")
    df, rename_map = standardize_column_names(df)
    report["column_renames"] = rename_map

    # 2. Normalize whitespace in text columns (before type detection)
    _step(1, "文本规范化")
    df = normalize_text_columns(df)

    # 3. Remove duplicates
    _step(2, "移除重复行")
    df, n_duplicates = remove_duplicates(df)
    report["duplicates_removed"] = n_duplicates

    # 4. Smart type detection & conversion
    _step(3, "智能类型检测与转换")
    df, type_report = detect_and_convert_types(df)
    report["type_conversions"] = type_report

    # 5. Re-normalize text columns (post type-conversion, for any new object cols)
    _step(4, "二次文本规范化")
    df = normalize_text_columns(df)

    # 6. Handle missing values
    _step(5, "缺失值处理")
    df, missing_report = handle_missing_values(
        df, numeric_strategy=numeric_strategy, categorical_strategy=categorical_strategy
    )
    report["missing_values"] = missing_report

    # 7. Handle outliers
    _step(6, "异常值处理")
    df, outlier_report = handle_outliers(df, method=outlier_method, iqr_multiplier=iqr_multiplier)
    report["outliers"] = outlier_report

    # 8. Optional: Data masking
    if enable_masking:
        _step(7, "数据脱敏")
        mask_kwargs = {}
        if masking_categories:
            mask_kwargs = masking_categories
        df, masking_report = mask_sensitive_data(
            df, columns=masking_columns, auto_detect=(masking_columns is None),
            **mask_kwargs,
        )
        report["masking"] = masking_report

    report["original_shape"] = original_shape
    report["final_shape"] = df.shape

    return df, report
