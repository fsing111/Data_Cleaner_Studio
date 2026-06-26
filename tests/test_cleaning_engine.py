"""
Tests for the cleaning engine module.

Run with:
    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from cleaning_engine import (
    standardize_column_names,
    remove_duplicates,
    detect_and_convert_types,
    handle_missing_values,
    handle_outliers,
    normalize_text_columns,
    detect_sensitive_columns,
    mask_sensitive_data,
    clean_dataframe,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Basic DataFrame with various column types and intentional messiness."""
    return pd.DataFrame({
        "  Employee Name  ": ["Alice", "Bob", "Charlie", "Alice", "David"],
        "Salary ($)": ["$50,000", "60,000", "55000", "$50,000", "(20000)"],
        "Join Date": ["2021-01-15", "01/20/2021", "2021-03-10", "2021-01-15", "March 5, 2022"],
        "Is Active?": ["Yes", "No", "yes", "Yes", "TRUE"],
        "Department": ["Sales", "IT", "Sales", "Sales", "HR"],
        "Age": [25, 30, 200, 25, 28],
        "Performance Rating": [4.5, 3.8, 4.1, 4.5, 3.9],
    })


@pytest.fixture
def df_with_missing():
    """DataFrame with missing values in various columns."""
    return pd.DataFrame({
        "name": ["Alice", None, "Charlie", "David", None],
        "score": [85.0, 92.0, None, 78.0, None],
        "category": pd.Categorical(["A", None, "B", "A", None]),
        "active": [True, False, None, True, None],
        "date": pd.to_datetime(["2021-01-01", None, "2021-03-01", None, "2021-05-01"]),
        "mostly_empty": [None, None, "only_value", None, None],
    })


@pytest.fixture
def df_with_outliers():
    """DataFrame with clear numeric outliers."""
    rng = np.random.default_rng(42)
    normal = rng.normal(50, 10, 98)
    outliers_high = np.full(1, 200)
    outliers_low = np.full(1, -50)
    all_vals = np.concatenate([normal, outliers_high, outliers_low])
    return pd.DataFrame({
        "value": all_vals,
        "category": ["A"] * 50 + ["B"] * 50,
    })


@pytest.fixture
def df_sensitive():
    """DataFrame with sensitive personal information."""
    return pd.DataFrame({
        "employee_name": ["张三", "Alice Wang", "李四", "Bob"],
        "phone": ["13812345678", "15900001111", "18612340000", "13099998888"],
        "email": ["zhangsan@company.com", "alice@email.com", "lisi@corp.cn", "bob@example.org"],
        "id_card": [
            "110101199001011234",
            "31010519851215002X",
            "440106197808080099",
            "500108200012310001",
        ],
        "department": ["Sales", "IT", "HR", "Finance"],
    })


# ── Tests: standardize_column_names ───────────────────────────────────────

class TestStandardizeColumnNames:
    def test_basic_standardization(self):
        df = pd.DataFrame({"  Employee Name  ": [1], "Salary ($)": [2], "Is Active?": [3]})
        result, rename_map = standardize_column_names(df)
        assert list(result.columns) == ["employee_name", "salary", "is_active"]

    def test_duplicate_names(self):
        # Create a DataFrame where two different column names map to the same standardized name
        df = pd.DataFrame({"a b": [1], "a_b": [2]})
        result, rename_map = standardize_column_names(df)
        # Both "a b" and "a_b" become "a_b", the duplicate gets numbered
        assert "a_b" in result.columns or "a_b_1" in result.columns
        assert len(result.columns) == 2

    def test_empty_column_name(self):
        df = pd.DataFrame({"": [1]})
        result, rename_map = standardize_column_names(df)
        assert result.columns[0] == "unnamed_column"

    def test_special_characters(self):
        df = pd.DataFrame({"col@#1": [1], "col$%^2": [2]})
        result, _ = standardize_column_names(df)
        assert "col1" in result.columns
        assert "col2" in result.columns

    def test_multiple_spaces(self):
        df = pd.DataFrame({"a   b   c": [1]})
        result, _ = standardize_column_names(df)
        assert "a_b_c" in result.columns


# ── Tests: remove_duplicates ──────────────────────────────────────────────

class TestRemoveDuplicates:
    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result, removed = remove_duplicates(df)
        assert len(result) == 3
        assert removed == 0

    def test_with_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result, removed = remove_duplicates(df)
        assert len(result) == 2
        assert removed == 1

    def test_all_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})
        result, removed = remove_duplicates(df)
        assert len(result) == 1
        assert removed == 2

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result, removed = remove_duplicates(df)
        assert len(result) == 0
        assert removed == 0


# ── Tests: detect_and_convert_types ───────────────────────────────────────

class TestDetectAndConvertTypes:
    def test_currency_to_numeric(self):
        df = pd.DataFrame({"salary": ["$50,000", "€60,000", "70,000"]})
        result, report = detect_and_convert_types(df)
        assert "numeric" in report["salary"]["converted_to"] or "float" in report["salary"]["converted_to"] or "int" in report["salary"]["converted_to"]

    def test_date_conversion(self):
        df = pd.DataFrame({"date": ["2021-01-15", "01/20/2021", "March 5, 2022"]})
        result, report = detect_and_convert_types(df)
        assert "datetime" in report["date"]["converted_to"]

    def test_boolean_conversion(self):
        df = pd.DataFrame({"flag": ["Yes", "No", "Yes", "No", "Yes"]})
        result, report = detect_and_convert_types(df)
        assert "bool" in report["flag"]["converted_to"].lower()

    def test_categorical_conversion(self):
        df = pd.DataFrame({"dept": ["Sales", "IT", "Sales", "HR", "IT"] * 5})
        result, report = detect_and_convert_types(df)
        assert report["dept"]["converted_to"] == "category"

    def test_already_numeric_preserved(self):
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
        result, report = detect_and_convert_types(df)
        assert "already numeric" in report["num"]["note"]

    def test_mixed_content_left_as_text(self):
        # Less than 80% numeric — should stay as object
        df = pd.DataFrame({"mixed": ["hello", "world", "42", "foo", "bar"]})
        result, report = detect_and_convert_types(df)
        assert "free text" in report["mixed"]["note"]

    def test_accounting_negative(self):
        df = pd.DataFrame({"amount": ["(500)", "(1,000)", "200"]})
        result, report = detect_and_convert_types(df)
        converted = report["amount"]["converted_to"]
        assert "numeric" in converted or "float" in converted or "int" in converted


# ── Tests: handle_missing_values ──────────────────────────────────────────

class TestHandleMissingValues:
    def test_numeric_median_fill(self, df_with_missing):
        result, report = handle_missing_values(df_with_missing, numeric_strategy="median")
        assert result["score"].isna().sum() == 0
        assert "filled with median" in report["score"]["action"]

    def test_numeric_mean_fill(self, df_with_missing):
        result, report = handle_missing_values(df_with_missing, numeric_strategy="mean")
        assert result["score"].isna().sum() == 0

    def test_numeric_zero_fill(self, df_with_missing):
        result, report = handle_missing_values(df_with_missing, numeric_strategy="zero")
        assert result["score"].isna().sum() == 0

    def test_categorical_mode_fill(self, df_with_missing):
        result, report = handle_missing_values(df_with_missing, categorical_strategy="mode")
        assert result["category"].isna().sum() == 0

    def test_categorical_missing_label(self, df_with_missing):
        result, report = handle_missing_values(df_with_missing, categorical_strategy="missing_label")
        assert result["category"].isna().sum() == 0
        assert "Missing" in result["category"].astype(str).values

    def test_boolean_mode_fill(self, df_with_missing):
        result, report = handle_missing_values(df_with_missing)
        assert result["active"].isna().sum() == 0

    def test_high_missing_column_dropped(self, df_with_missing):
        result, report = handle_missing_values(df_with_missing)
        assert "mostly_empty" not in result.columns
        assert "dropped column" in report["mostly_empty"]["action"]

    def test_no_missing_values(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result, report = handle_missing_values(df)
        assert len(report) == 0
        assert len(result) == 3


# ── Tests: handle_outliers ────────────────────────────────────────────────

class TestHandleOutliers:
    def test_cap_outliers(self, df_with_outliers):
        result, report = handle_outliers(df_with_outliers, method="cap", iqr_multiplier=1.5)
        assert result["value"].max() <= df_with_outliers["value"].quantile(0.75) + 1.5 * (
            df_with_outliers["value"].quantile(0.75) - df_with_outliers["value"].quantile(0.25)
        )
        assert "capped" in report["value"]["action"]

    def test_remove_outliers(self, df_with_outliers):
        result, report = handle_outliers(df_with_outliers, method="remove", iqr_multiplier=1.5)
        assert len(result) < len(df_with_outliers)

    def test_detect_only(self, df_with_outliers):
        result, report = handle_outliers(df_with_outliers, method="none", iqr_multiplier=1.5)
        assert len(result) == len(df_with_outliers)
        assert "no action taken" in report["value"]["action"]

    def test_no_outliers(self):
        df = pd.DataFrame({"a": [10, 11, 12, 13, 14]})
        result, report = handle_outliers(df)
        assert len(report) == 0

    def test_binary_column_skipped(self):
        df = pd.DataFrame({"flag": [0, 1, 0, 1, 0], "value": [10, 11, 12, 13, 14]})
        result, report = handle_outliers(df)
        # flag has 2 unique values — should be skipped
        assert "flag" not in report

    def test_iqr_multiplier_sensitivity(self):
        df = pd.DataFrame({"a": [10, 11, 12, 13, 100]})
        # With very high multiplier, outlier may not be detected
        _, report_strict = handle_outliers(df, iqr_multiplier=1.0)
        _, report_loose = handle_outliers(df.copy(), iqr_multiplier=3.0)
        # Stricter multiplier detects more
        n_strict = report_strict.get("a", {}).get("n_outliers", 0)
        n_loose = report_loose.get("a", {}).get("n_outliers", 0)
        assert n_strict >= n_loose


# ── Tests: normalize_text_columns ─────────────────────────────────────────

class TestNormalizeTextColumns:
    def test_strip_whitespace(self):
        df = pd.DataFrame({"text": ["  hello  ", "  world", "foo  "]})
        result = normalize_text_columns(df)
        assert result["text"].iloc[0] == "hello"
        assert result["text"].iloc[1] == "world"

    def test_collapse_multi_spaces(self):
        df = pd.DataFrame({"text": ["hello    world", "foo   bar   baz"]})
        result = normalize_text_columns(df)
        assert result["text"].iloc[0] == "hello world"

    def test_nan_handling(self):
        df = pd.DataFrame({"text": ["hello", "nan", "None", ""]})
        result = normalize_text_columns(df)
        assert pd.isna(result["text"].iloc[1])
        assert pd.isna(result["text"].iloc[2])
        assert pd.isna(result["text"].iloc[3])

    def test_numeric_columns_unchanged(self):
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
        result = normalize_text_columns(df)
        assert list(result["num"]) == [1.0, 2.0, 3.0]


# ── Tests: detect_sensitive_columns ───────────────────────────────────────

class TestDetectSensitiveColumns:
    def test_detect_by_column_name(self):
        df = pd.DataFrame({
            "employee_name": ["Alice"],
            "phone_number": ["123"],
            "email_address": ["a@b.com"],
        })
        result = detect_sensitive_columns(df)
        assert "name" in result
        assert "phone" in result
        assert "email" in result
        assert "employee_name" in result["name"]
        assert "phone_number" in result["phone"]

    def test_detect_by_content(self):
        df = pd.DataFrame({
            "col_a": ["13812345678", "15900001111", "18612340000"],
            "col_b": ["user@mail.com", "test@test.org", "foo@bar.cn"],
        })
        result = detect_sensitive_columns(df)
        assert "phone" in result
        assert "col_a" in result["phone"]
        assert "email" in result
        assert "col_b" in result["email"]

    def test_chinese_keywords(self):
        df = pd.DataFrame({
            "姓名": ["张三"],
            "电话": ["138"],
            "邮箱": ["a@b.com"],
        })
        result = detect_sensitive_columns(df)
        assert "姓名" in result.get("name", [])
        assert "电话" in result.get("phone", [])
        assert "邮箱" in result.get("email", [])

    def test_no_sensitive_data(self):
        df = pd.DataFrame({
            "department": ["Sales", "IT"],
            "score": [85, 92],
        })
        result = detect_sensitive_columns(df)
        assert result == {}


# ── Tests: mask_sensitive_data ────────────────────────────────────────────

class TestMaskSensitiveData:
    def test_mask_chinese_name(self, df_sensitive):
        result, report = mask_sensitive_data(df_sensitive, auto_detect=True)
        # 张三 -> 张*
        masked_name = result.loc[0, "employee_name"]
        assert masked_name == "张*"
        # Alice Wang -> A****
        masked_name2 = result.loc[1, "employee_name"]
        assert masked_name2.startswith("A")

    def test_mask_phone(self, df_sensitive):
        result, report = mask_sensitive_data(df_sensitive, auto_detect=True)
        # 13812345678 -> 138****5678
        assert "****" in result.loc[0, "phone"]
        assert result.loc[0, "phone"].startswith("138")
        assert result.loc[0, "phone"].endswith("5678")

    def test_mask_email(self, df_sensitive):
        result, report = mask_sensitive_data(df_sensitive, auto_detect=True)
        # zhangsan@company.com -> z****@company.com
        masked = result.loc[0, "email"]
        assert "****" in masked
        assert "@company.com" in masked

    def test_mask_id_card(self, df_sensitive):
        result, report = mask_sensitive_data(df_sensitive, auto_detect=True)
        # 110101199001011234 -> 110101********1234
        masked = result.loc[0, "id_card"]
        assert "********" in masked
        assert masked.startswith("110101")
        assert masked.endswith("1234")

    def test_department_not_masked(self, df_sensitive):
        result, report = mask_sensitive_data(df_sensitive, auto_detect=True)
        assert result.loc[0, "department"] == "Sales"

    def test_manual_columns(self, df_sensitive):
        result, report = mask_sensitive_data(
            df_sensitive,
            columns={"phone": ["phone"]},
            auto_detect=False,
            mask_name=False,
            mask_phone=True,
            mask_email=False,
            mask_id_card=False,
            mask_address=False,
        )
        # Only phone should be masked
        assert "****" in result.loc[0, "phone"]
        assert result.loc[0, "employee_name"] == "张三"
        assert result.loc[0, "email"] == "zhangsan@company.com"

    def test_nan_handling(self):
        df = pd.DataFrame({
            "phone": ["13812345678", None, "", "15900001111"],
        })
        result, report = mask_sensitive_data(df, columns={"phone": ["phone"]}, auto_detect=False)
        assert pd.isna(result.loc[1, "phone"])
        assert result.loc[0, "phone"] != "13812345678"


# ── Tests: clean_dataframe (integration) ──────────────────────────────────

class TestCleanDataFrame:
    def test_full_pipeline(self, sample_df):
        result, report = clean_dataframe(sample_df)
        assert "original_shape" in report
        assert "final_shape" in report
        assert "column_renames" in report
        assert "duplicates_removed" in report
        assert "type_conversions" in report
        assert "missing_values" in report
        assert "outliers" in report

    def test_duplicates_removed(self, sample_df):
        result, report = clean_dataframe(sample_df)
        # Alice appears twice in sample_df
        assert report["duplicates_removed"] >= 1

    def test_progress_callback(self, sample_df):
        steps = []

        def cb(idx, total, name):
            steps.append((idx, total, name))

        result, report = clean_dataframe(sample_df, progress_callback=cb)
        assert len(steps) == 7
        assert steps[0] == (0, 7, "标准化列名")
        assert steps[-1] == (6, 7, "异常值处理")

    def test_progress_callback_with_masking(self, sample_df):
        steps = []

        def cb(idx, total, name):
            steps.append((idx, total, name))

        result, report = clean_dataframe(sample_df, progress_callback=cb, enable_masking=True)
        assert len(steps) == 8
        assert steps[-1] == (7, 8, "数据脱敏")

    def test_masking_integration(self, df_sensitive):
        result, report = clean_dataframe(df_sensitive, enable_masking=True)
        assert "masking" in report
        assert report["masking"]["total_masked"] > 0
        # Names should be masked
        assert "张*" in result["employee_name"].values or any(
            "*" in str(v) for v in result["employee_name"].dropna()
        )

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result, report = clean_dataframe(df)
        assert len(result) == 0
        assert report["original_shape"] == (0, 0)
        assert report["final_shape"] == (0, 0)

    def test_single_row(self):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        result, report = clean_dataframe(df)
        assert len(result) == 1

    def test_outlier_methods(self, sample_df):
        # All three methods should complete without error
        for method in ["cap", "remove", "none"]:
            result, report = clean_dataframe(sample_df.copy(), outlier_method=method)
            assert "outliers" in report

    def test_numeric_strategies(self, sample_df):
        for strategy in ["median", "mean", "zero"]:
            result, report = clean_dataframe(sample_df.copy(), numeric_strategy=strategy)
            assert "missing_values" in report
