"""DeepSeek API 集成模块 — LLM 对话式数据问答。

使用 OpenAI 兼容接口调用 DeepSeek API，为清洗后的数据提供
自然语言问答能力。
"""

from __future__ import annotations

import os
from typing import Generator

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# DeepSeek API 配置
# ---------------------------------------------------------------------------

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

# ---------------------------------------------------------------------------
# 数据上下文构建
# ---------------------------------------------------------------------------


def build_data_context(
    df: pd.DataFrame,
    report: dict,
    max_sample_rows: int = 20,
) -> str:
    """从清洗后的 DataFrame 和质量报告构建结构化文字上下文。

    上下文会作为 system message 的一部分注入给 LLM，
    使其能够回答关于这份特定数据的问题。

    Parameters
    ----------
    df : pd.DataFrame
        清洗后的数据。
    report : dict
        cleaning_engine.clean_dataframe 返回的清洗报告。
    max_sample_rows : int
        采样展示的最大行数。

    Returns
    -------
    str
        Markdown 格式的数据上下文。
    """
    rows, cols = df.shape

    parts: list[str] = []

    # --- 基本信息 ---
    parts.append("## 数据集基本信息")
    parts.append(f"- 总行数：{rows}")
    parts.append(f"- 总列数：{cols}")
    parts.append("")

    # --- 列信息 ---
    parts.append("## 列名称与数据类型")
    parts.append("| 列名 | 数据类型 |")
    parts.append("|------|----------|")
    for col_name in df.columns:
        parts.append(f"| {col_name} | {df[col_name].dtype} |")
    parts.append("")

    # --- 描述性统计（数值列） ---
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        parts.append("## 数值列描述性统计")
        try:
            desc = df[numeric_cols].describe().to_string()
            parts.append("```")
            parts.append(desc)
            parts.append("```")
        except Exception:
            pass
        parts.append("")

    # --- 分类列概览 ---
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if cat_cols:
        parts.append("## 文本/分类列概览")
        for c in cat_cols[:10]:  # 最多展示 10 列
            unique_count = df[c].nunique()
            missing_count = df[c].isna().sum()
            parts.append(f"- **{c}**：{unique_count} 个唯一值，{missing_count} 个缺失值")
        parts.append("")

    # --- 质量评分 ---
    quality = report.get("quality_score")
    if quality is not None:
        parts.append("## 数据质量评分")
        parts.append(f"- 综合质量评分：**{quality}/100**")
    completeness = report.get("cleaned_completeness")
    if completeness is not None:
        parts.append(f"- 完整性：{completeness}%")
    dup_rate = report.get("cleaned_dup_rate")
    if dup_rate is not None:
        parts.append(f"- 重复率：{dup_rate}%")
    parts.append("")

    # --- 类型转换 ---
    type_conversions = report.get("type_conversions", {})
    converted = {
        k: v for k, v in type_conversions.items()
        if "already" not in v.get("note", "") and "free text" not in v.get("note", "")
    }
    if converted:
        parts.append("## 已转换的数据类型")
        for col_name, info in converted.items():
            parts.append(f"- **{col_name}**：{info['original']} → {info['converted_to']}（{info['note']}）")
        parts.append("")

    # --- 异常值 ---
    outliers = report.get("outliers", {})
    if outliers:
        parts.append("## 检测到的异常值")
        for col_name, info in outliers.items():
            parts.append(
                f"- **{col_name}**：{info['n_outliers']} 个异常值，"
                f"范围 [{info['bounds'][0]}, {info['bounds'][1]}]，"
                f"处理方式：{info['action']}"
            )
        parts.append("")

    # --- 缺失值 ---
    missing = report.get("missing_values", {})
    if missing:
        parts.append("## 缺失值处理")
        for col_name, info in missing.items():
            parts.append(f"- **{col_name}**：{info['missing_pct']}% 缺失 → {info['action']}")
        parts.append("")

    # --- 数据样本 ---
    parts.append(f"## 数据样本（前 {min(max_sample_rows, rows)} 行）")
    parts.append("```")
    try:
        parts.append(df.head(max_sample_rows).to_string())
    except Exception:
        parts.append("（无法生成样本）")
    parts.append("```")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# DeepSeek 客户端
# ---------------------------------------------------------------------------


def get_api_key(sidebar_value: str = "") -> str | None:
    """按优先级获取 DeepSeek API Key。

    优先级：侧边栏输入 > Streamlit Secrets > 环境变量 > .env 文件

    Parameters
    ----------
    sidebar_value : str
        用户在侧边栏输入的 API Key。

    Returns
    -------
    str or None
        API Key，如果所有来源都没有则返回 None。
    """
    # 1. 侧边栏输入
    if sidebar_value and sidebar_value.strip():
        return sidebar_value.strip()

    # 2. Streamlit Secrets
    try:
        secret_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if secret_key and secret_key.strip():
            return secret_key.strip()
    except Exception:
        pass

    # 3. 环境变量
    env_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if env_key and env_key.strip():
        return env_key.strip()

    # 4. 尝试从 .env 文件加载（如果 python-dotenv 可用）
    try:
        from dotenv import load_dotenv
        load_dotenv()
        env_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if env_key and env_key.strip():
            return env_key.strip()
    except ImportError:
        pass

    return None


def create_client(api_key: str) -> OpenAI:
    """创建指向 DeepSeek API 的 OpenAI 客户端。

    Parameters
    ----------
    api_key : str
        DeepSeek API Key。

    Returns
    -------
    OpenAI
    """
    return OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
    )


def test_connection(api_key: str) -> tuple[bool, str]:
    """测试 DeepSeek API 连接。

    Parameters
    ----------
    api_key : str
        待测试的 API Key。

    Returns
    -------
    tuple[bool, str]
        (是否成功, 消息)
    """
    if not api_key or not api_key.strip():
        return False, "请先输入 API Key"

    try:
        client = create_client(api_key.strip())
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
            stream=False,
        )
        reply = response.choices[0].message.content
        return True, f"✅ 连接成功！DeepSeek 响应：{reply}"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Authentication" in error_msg:
            return False, "❌ API Key 无效，请检查后重试"
        elif "402" in error_msg or "Insufficient" in error_msg:
            return False, "❌ 账户余额不足，请充值"
        elif "429" in error_msg:
            return False, "❌ 请求过于频繁，请稍后重试"
        else:
            return False, f"❌ 连接失败：{error_msg}"


# ---------------------------------------------------------------------------
# 聊天
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
你是一位专业的数据分析师，你的任务是帮助用户理解和分析他们上传的数据。

## 角色要求
- 使用中文回答（除非用户用英文提问）。
- 回答要专业、准确、简洁。
- 如果用户的问题与数据无关，礼貌地引导他们回到数据分析主题。
- 当用户问"这个数据有什么问题"、"数据质量如何"等，结合上下文中的质量评分和清洗报告给出具体分析。
- 可以帮用户做统计分析、发现数据模式、解释数据特征、给出改进建议。

## 当前数据集上下文
{data_context}

请基于以上上下文回答用户的问题。"""


def chat_with_data(
    client: OpenAI,
    messages: list[dict],
    data_context: str,
    model: str = DEFAULT_MODEL,
) -> Generator[str, None, None]:
    """流式调用 DeepSeek API 进行数据问答。

    Parameters
    ----------
    client : OpenAI
        已配置的 DeepSeek 客户端。
    messages : list[dict]
        历史消息列表，每条含 "role" 和 "content"。
        函数会自动在前面插入系统提示词。
    data_context : str
        数据上下文文本。
    model : str
        模型名称。

    Yields
    ------
    str
        每次 yield 一个文本片段。
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(data_context=data_context)

    full_messages = [
        {"role": "system", "content": system_prompt},
        *messages,
    ]

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=full_messages,
            stream=True,
            max_tokens=4096,
            temperature=0.7,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    except Exception as e:
        yield f"\n\n⚠️ 调用 DeepSeek API 时出错：{e}"
