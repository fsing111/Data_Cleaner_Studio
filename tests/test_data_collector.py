"""
Tests for the data collector module.

Run with:
    pytest tests/ -v
"""

import json
import os
import tempfile

import pandas as pd
import pytest

from data_collector import (
    scrape_tables,
    fetch_api_data,
    load_file,
    collect_and_save,
    _guess_source_type,
    _check_compliance,
)


# ══════════════════════════════════════════════════════════════════════════════
# 单元测试：合规检查 & 来源推断
# ══════════════════════════════════════════════════════════════════════════════

class TestCompliance:
    """合规性检查。"""

    def test_valid_https_url(self):
        _check_compliance("https://example.com/data.html")  # 不应抛异常

    def test_valid_http_url(self):
        _check_compliance("http://example.com/data.html")  # 不应抛异常

    def test_invalid_ftp_url(self):
        with pytest.raises(ValueError, match="不支持的协议"):
            _check_compliance("ftp://example.com/files")


class TestGuessSourceType:
    """来源类型推断。"""

    def test_csv_file(self):
        assert _guess_source_type("https://example.com/data.csv") == "file"

    def test_xlsx_file(self):
        assert _guess_source_type("https://example.com/data.xlsx") == "file"

    def test_api_endpoint(self):
        assert _guess_source_type("https://api.example.com/v1/records") == "api"

    def test_json_endpoint(self):
        assert _guess_source_type("https://example.com/data/json") == "api"

    def test_default_table(self):
        assert _guess_source_type("https://en.wikipedia.org/wiki/Python") == "table"


# ══════════════════════════════════════════════════════════════════════════════
# 集成测试：文件采集
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadFile:
    """本地文件加载。"""

    def test_load_csv(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("name,age\nAlice,25\nBob,30\n")
            path = f.name

        try:
            df = load_file(path)
            assert len(df) == 2
            assert list(df.columns) == ["name", "age"]
            assert df.iloc[0]["name"] == "Alice"
        finally:
            os.unlink(path)

    def test_load_xlsx(self):
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            path = f.name

        try:
            df_src = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
            df_src.to_excel(path, index=False)
            df = load_file(path)
            assert len(df) == 2
            assert list(df.columns) == ["col_a", "col_b"]
        finally:
            os.unlink(path)


# ══════════════════════════════════════════════════════════════════════════════
# 集成测试：collect_and_save
# ══════════════════════════════════════════════════════════════════════════════

class TestCollectAndSave:
    """一站式采集流水线。"""

    def test_collect_from_file_and_save(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("x,y\n1,a\n2,b\n3,c\n")
            src_path = f.name

        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False
        ) as f:
            out_path = f.name

        try:
            df = collect_and_save(src_path, out_path, source_type="file")
            assert len(df) == 3
            assert os.path.exists(out_path)

            # 验证输出文件内容
            df_out = pd.read_csv(out_path)
            assert len(df_out) == 3
        finally:
            os.unlink(src_path)
            if os.path.exists(out_path):
                os.unlink(out_path)
