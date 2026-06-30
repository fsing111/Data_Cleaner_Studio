# 自动化数据清洗平台

上传 CSV 或 Excel 文件，自动完成列名标准化、类型检测、去重、缺失值填补、异常值处理和数据脱敏，输出清洗后的文件、质量评分和完整报告。内置 DeepSeek 驱动的 AI 对话助手，可以直接用自然语言向数据提问。

## 快速开始

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开 `http://localhost:8501`，上传 `sample_messy_data.csv` 即可看到完整流程。

如果需要大量测试数据：

```bash
python generate_sample_data.py --rows 100000 --output big_data.csv
```

## 功能

### 列名标准化

`  Employee Name  `、`Salary ($)`、`Is Active?` 这类混乱列名统一转为小写 + 下划线格式（snake_case），去除特殊字符，合并多余空格，重名自动编号。

### 文本规范化

去除文本列首尾空格，合并内部连续空格（`"  good  "` → `"good"`）。同时将字符串 `"nan"`、`"None"`、空字符串识别为缺失值。

### 重复行删除

删除完全重复的行，记录删除数量。

### 智能类型检测与转换

对文本列按优先级尝试类型转换，只有 ≥80% 非空值解析成功时才执行：

| 目标类型 | 能处理的格式 |
|----------|-------------|
| 数值 | 货币符号（`$50,000`）、千分位、百分号、会计负数 `(20000)` |
| 日期时间 | 混合日期格式（`2021-01-15`、`01/20/2021`、`March 5, 2022`） |
| 布尔值 | Yes/No、True/False、Y/N、1/0 |
| 分类 | 低基数文本（不超过 50 个唯一值、占比不超过 50%） |
| 文本 | 以上都不匹配则保持字符串 |

### 缺失值处理

- 数值列：中位数（默认）、均值、或填零
- 日期列：线性插值，辅以前向/后向填充
- 布尔列：众数填充
- 分类/文本列：众数填充，或标记为 `"Missing"`
- 缺失超过 60% 的列直接删除

### 异常值处理

基于 IQR（四分位距）检测数值列异常值，跳过只有 0/1 两值的标记列：

- **截尾**：值裁剪到 `[Q1 − k×IQR, Q3 + k×IQR]` 范围内（默认 k=1.5）
- **删除**：移除包含异常值的行
- **仅检测**：只记录不处理

### 数据脱敏

通过列名和内容模式自动识别敏感列，支持中文关键词（姓名、手机、邮箱、身份证、地址）：

| 类别 | 效果 |
|------|------|
| 姓名 | `张三` → `张*`，`Alice Wang` → `A****` |
| 手机号 | `13812345678` → `138****5678` |
| 身份证号 | `110101199001011234` → `110101********1234` |
| 邮箱 | `zhangsan@company.com` → `z****@company.com` |
| 地址 | `北京市朝阳区中山路100号` → `北京市朝阳区******` |

每种脱敏类别可以在侧边栏单独开关。

### 数据质量评分

综合三个维度计算 0-100 分：完整度（50%）、唯一性（20%）、类型一致性（30%）。清洗报告包含清洗前后的完整度对比、重复率变化、数值列统计摘要。

### 可视化

- 缺失值柱状图：清洗前后并排对比
- 异常值箱线图：每个有异常值的数值列处理前后对比
- 分布直方图：异常值处理前后叠加展示

### AI 数据助手

基于 DeepSeek API 的对话式数据问答。上传数据并完成清洗后，切换到 AI 助手标签页，用自然语言向数据提问，比如"薪资列的分布情况如何""哪个部门平均绩效最高""数据有哪些质量问题"。回答会结合当前数据的结构、统计信息和清洗报告实时生成。

使用前需要在侧边栏的 AI 设置中输入 DeepSeek API Key（[获取地址](https://platform.deepseek.com/api_keys)），也可以配置在 `.env` 文件或 Streamlit Secrets 中。

### Excel 支持

支持 `.csv`、`.xlsx`、`.xls` 格式导入，清洗结果可下载为 CSV 或 Excel。

## 输出

- 清洗前后对比预览
- 标签页报告：质量评分、可视化图表、类型转换明细、缺失值处理记录、异常值信息、脱敏摘要、列名变更
- CSV / Excel 双格式下载
- JSON 完整清洗报告

## Streamlit Cloud 部署

1. 将代码推送到 GitHub
2. 登录 [share.streamlit.io](https://share.streamlit.io)，用 GitHub 账号注册
3. 点击 "Create App"，选择仓库和分支，主文件路径填 `app.py`
4. 在 "Advanced settings" → "Secrets" 中配置 API Key：

```toml
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxx"
```

5. 点击 "Deploy"，等待构建完成即可

本地开发时，复制 `.env.example` 为 `.env` 并填入真实 Key，或将 Key 写在 `.streamlit/secrets.toml` 中（该文件不会被提交到 Git）。

## 生成测试数据

```bash
# 1 万行，默认 10% 噪声
python generate_sample_data.py

# 100 万行
python generate_sample_data.py --rows 1000000 --output million_rows.csv

# 自定义参数
python generate_sample_data.py --rows 50000 --noise 0.15 --seed 123
```

参数：

- `--rows`：生成行数，默认 10000
- `--output`：输出文件路径
- `--seed`：随机种子，保证结果可复现
- `--noise`：注入问题的行比例，默认 0.1

生成字段包括员工姓名、薪资、入职日期、是否在职、部门、年龄、绩效评级、手机号、邮箱、身份证号、地址，数据中刻意混入了格式不一致、缺失值、异常值和重复行。

## 测试

```bash
pytest tests/ -v
```

覆盖 `cleaning_engine.py` 全部核心逻辑：列名标准化、去重、类型检测、缺失值处理、异常值处理、敏感列检测与脱敏、完整流水线集成。

## 项目结构

```
AI训练数据智能清洗平台/
├── app.py                   # Streamlit 界面
├── cleaning_engine.py       # 清洗流水线（纯逻辑，可独立调用）
├── llm_advisor.py           # DeepSeek API 集成（AI 数据助手）
├── data_collector.py        # 网页表格抓取与 API 数据采集
├── generate_sample_data.py  # 脏数据生成器
├── requirements.txt
├── .env.example             # 环境变量模板
├── sample_messy_data.csv    # 示例文件
├── .streamlit/
│   ├── config.toml          # 部署配置
│   └── secrets.toml         # 本地密钥（不提交 Git）
└── tests/
    ├── __init__.py
    ├── test_cleaning_engine.py
    └── test_data_collector.py
```

`cleaning_engine.py` 可以脱离 Streamlit 单独使用：

```python
import pandas as pd
from cleaning_engine import clean_dataframe

df = pd.read_csv("my_data.csv")
cleaned_df, report = clean_dataframe(
    df,
    enable_masking=True,
    progress_callback=lambda i, t, n: print(f"[{i+1}/{t}] {n}"),
)
```

## 技术栈

- Python 3.10+
- Streamlit（Web 界面）
- Pandas（数据处理）
- NumPy（数值计算）
- Matplotlib（可视化）
- OpenPyXL（Excel 读写）
- OpenAI SDK（DeepSeek API 调用，OpenAI 兼容接口）
- pytest（测试）
