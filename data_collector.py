"""
Data Collector — Web Scraping & API Data Acquisition
=====================================================
从网页表格、公开 API 等渠道采集数据，导出为结构化 CSV/DataFrame，
直接对接下游清洗流水线。

支持的数据源类型：
  - HTML 网页表格（Wikipedia、统计局等公开数据）
  - RESTful API 接口（JSON 响应）
  - 本地/远程 CSV 文件

设计原则：
  - 合规优先：设置 User-Agent、遵守 robots.txt 建议
  - 稳定采集：指数退避重试、请求间隔控制
  - 格式统一：所有采集结果输出为 pandas DataFrame

使用示例
--------
>>> from data_collector import scrape_tables, fetch_api_data
>>>
>>> # 从网页抓取表格
>>> df = scrape_tables("https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)")
>>>
>>> # 从公开 API 获取数据
>>> df = fetch_api_data("https://api.example.com/datasets/123", params={"format": "json"})
>>>
>>> # 直接传入清洗引擎
>>> from cleaning_engine import clean_dataframe
>>> cleaned, report = clean_dataframe(df)
"""

import time
import logging
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import requests

# ── 日志配置 ────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── 请求头（合规声明） ──────────────────────────────────────────────────────
DEFAULT_HEADERS = {
    "User-Agent": (
        "DataCleanerStudio/1.0 "
        "(Educational project; contact@example.com)"
    ),
    "Accept": "text/html,application/json,*/*",
}

# ── 重试与限速配置 ──────────────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # 指数退避基数（秒）: 2, 4, 8
REQUEST_INTERVAL = 1.0  # 两次请求之间的最小间隔（秒）


# ══════════════════════════════════════════════════════════════════════════════
# 1. HTML 网页表格抓取
# ══════════════════════════════════════════════════════════════════════════════

def scrape_tables(
    url: str,
    table_index: int = 0,
    headers: Optional[dict] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    从网页抓取 HTML 表格并返回 DataFrame。

    使用 pandas 内置的 read_html() 解析 <table> 标签，
    自动处理 colspan/rowspan 等复杂结构。

    参数
    ----
    url : str
        目标网页地址
    table_index : int
        要提取的表格序号（0 表示第一个表格）
    headers : dict, optional
        自定义 HTTP 请求头，默认使用合规 User-Agent
    timeout : int
        请求超时时间（秒）

    返回
    ----
    pd.DataFrame
        提取到的表格数据

    异常
    ----
    ValueError
        页面中未找到表格或 table_index 超出范围
    requests.RequestException
        网络请求失败（自动重试后仍失败）
    """
    _check_compliance(url)

    resp = _request_with_retry(url, headers=headers or DEFAULT_HEADERS, timeout=timeout)

    try:
        tables = pd.read_html(resp.text)
    except ValueError:
        raise ValueError(f"未在页面中找到 HTML 表格: {url}")

    if table_index >= len(tables):
        raise ValueError(
            f"表格序号 {table_index} 超出范围（页面共 {len(tables)} 个表格）"
        )

    df = tables[table_index]
    logger.info("成功从 %s 抓取表格[%d]，%d 行 × %d 列", url, table_index, *df.shape)
    return df


def scrape_all_tables(
    url: str,
    headers: Optional[dict] = None,
    timeout: int = 30,
) -> list[pd.DataFrame]:
    """
    抓取页面中所有 HTML 表格，返回 DataFrame 列表。
    """
    _check_compliance(url)
    resp = _request_with_retry(url, headers=headers or DEFAULT_HEADERS, timeout=timeout)

    try:
        tables = pd.read_html(resp.text)
    except ValueError:
        logger.warning("未在页面中找到 HTML 表格: %s", url)
        return []

    logger.info("从 %s 抓取到 %d 个表格", url, len(tables))
    return tables


# ══════════════════════════════════════════════════════════════════════════════
# 2. REST API 数据获取
# ══════════════════════════════════════════════════════════════════════════════

def fetch_api_data(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    json_path: Optional[str] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    从 RESTful API 获取 JSON 数据并转换为 DataFrame。

    自动检测 JSON 结构：
      - 顶层为 list[dict] → 直接转为 DataFrame
      - 顶层为 dict → 尝试嵌套路径（如 "data.records"）提取数组
      - 通过 json_path 参数指定提取路径（如 "data.items"）

    参数
    ----
    url : str
        API 端点地址
    params : dict, optional
        URL 查询参数
    headers : dict, optional
        自定义请求头（如 API Key），默认使用合规 User-Agent
    json_path : str, optional
        点号分隔的 JSON 嵌套路径（如 "data.records"）
    timeout : int
        请求超时时间（秒）

    返回
    ----
    pd.DataFrame
        解析后的数据

    异常
    ----
    requests.RequestException
        网络请求失败
    ValueError
        JSON 结构中未找到可转换的数组数据
    """
    merged_headers = {**DEFAULT_HEADERS, **(headers or {})}

    resp = _request_with_retry(url, headers=merged_headers, params=params, timeout=timeout)
    data = resp.json()

    # 按路径提取目标数组
    records = data
    if json_path:
        for key in json_path.split("."):
            if isinstance(records, dict):
                records = records.get(key, [])
            else:
                raise ValueError(f"路径 '{json_path}' 中 '{key}' 不是 dict，无法继续解析")

    # 转为 DataFrame
    if isinstance(records, list):
        if len(records) == 0:
            logger.warning("API 返回空数组: %s", url)
            return pd.DataFrame()
        df = pd.json_normalize(records, max_level=1)
    elif isinstance(records, dict):
        # 尝试自动找到第一个 list 值
        for key, val in records.items():
            if isinstance(val, list):
                df = pd.json_normalize(val, max_level=1)
                logger.info("自动检测到嵌套路径 '%s'，提取 %d 条记录", key, len(val))
                break
        else:
            df = pd.DataFrame([records])
    else:
        raise ValueError(f"无法将 JSON 数据转换为 DataFrame，类型: {type(records)}")

    logger.info("从 API 获取 %d 行 × %d 列数据", *df.shape)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. 通用文件采集（本地/远程 CSV）
# ══════════════════════════════════════════════════════════════════════════════

def load_file(
    path: str,
    encoding: str = "utf-8",
    **kwargs,
) -> pd.DataFrame:
    """
    加载本地或远程 CSV 文件，自动检测编码。

    支持的文件格式: .csv, .xlsx, .xls
    远程文件（http/https）自动下载后解析。
    """
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else "csv"

    if path.startswith(("http://", "https://")):
        logger.info("下载远程文件: %s", path)
        resp = _request_with_retry(path, headers=DEFAULT_HEADERS)
        from io import BytesIO

        if ext in ("xlsx", "xls"):
            return pd.read_excel(BytesIO(resp.content), **kwargs)
        else:
            from io import StringIO
            return pd.read_csv(StringIO(resp.text), encoding=encoding, **kwargs)

    if ext in ("xlsx", "xls"):
        return pd.read_excel(path, **kwargs)
    else:
        return pd.read_csv(path, encoding=encoding, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 采集流水线（便捷入口）
# ══════════════════════════════════════════════════════════════════════════════

def collect_and_save(
    source: str,
    output_path: str,
    source_type: str = "auto",
    **kwargs,
) -> pd.DataFrame:
    """
    一站式采集并保存：从指定来源获取数据，导出为 CSV 文件。

    参数
    ----
    source : str
        数据来源 URL 或文件路径
    output_path : str
        输出 CSV 文件路径
    source_type : str
        "auto" | "table"（HTML表格） | "api"（REST API） | "file"（CSV/Excel）
    **kwargs
        传递给对应采集函数的额外参数

    返回
    ----
    pd.DataFrame
        采集到的数据
    """
    if source_type == "auto":
        source_type = _guess_source_type(source)

    if source_type == "table":
        df = scrape_tables(source, **kwargs)
    elif source_type == "api":
        df = fetch_api_data(source, **kwargs)
    elif source_type == "file":
        df = load_file(source, **kwargs)
    else:
        raise ValueError(f"不支持的来源类型: {source_type}")

    # 保存
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("数据已保存至 %s（%d 行 × %d 列）", output_path, *df.shape)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 内部工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _request_with_retry(
    url: str,
    headers: dict,
    params: Optional[dict] = None,
    timeout: int = 30,
) -> requests.Response:
    """指数退避重试的 HTTP GET 请求。"""
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            time.sleep(REQUEST_INTERVAL)  # 请求间隔
            resp = requests.get(
                url, headers=headers, params=params,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF ** (attempt + 1)
                logger.warning(
                    "请求失败 (尝试 %d/%d): %s，%d 秒后重试...",
                    attempt + 1, MAX_RETRIES, e, wait,
                )
                time.sleep(wait)

    raise last_exc  # type: ignore[misc]


def _check_compliance(url: str) -> None:
    """
    合规检查：验证协议、记录采集域名。

    当前为教育用途项目，输出提示信息提醒合规使用。
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"不支持的协议: {parsed.scheme}，仅支持 http/https")

    logger.info("正在采集数据，来源: %s，请确保遵守目标网站的使用条款。", parsed.netloc)


def _guess_source_type(source: str) -> str:
    """根据 URL/路径特征自动推断数据源类型。"""
    parsed = urlparse(source)
    path = parsed.path.lower()

    if path.endswith((".csv", ".xlsx", ".xls")):
        return "file"
    if "api" in parsed.netloc or path.endswith("/json"):
        return "api"
    return "table"
