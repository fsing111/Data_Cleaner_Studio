# 🧹 自动化数据清洗平台

上传一份混乱的 CSV 或 Excel 文件，返回一份经过清洗、类型正确、可下载的数据集——附带完整的透明度报告、数据质量评分、可视化图表以及可选的数据脱敏功能。

---

## 快速开始

```bash
pip install -r requirements.txt
streamlit run app.py
```

在浏览器中打开 `http://localhost:8501`。上传 `sample_messy_data.csv` 即可查看完整示例。

如需大规模测试，可生成模拟脏数据：

```bash
python generate_sample_data.py --rows 100000 --output big_data.csv
```

---

## 功能说明

### 1. 列名标准化

`  Employee Name  ` → `employee_name`
`Salary ($)` → `salary`
`Is Active?` → `is_active`

转为小写、下划线命名（snake_case），去除特殊字符，合并多余空白，重名自动编号。

### 2. 空白与文本规范化

去除所有文本列的首尾空格，合并内部多余空格（`"  good  "` → `"good"`）。

### 3. 重复行删除

删除完全重复的行。

### 4. 智能类型检测与转换

对每个文本/对象列，按以下优先级尝试转换：

| 步骤 | 检测类型 | 示例 |
|---|---|---|
| 数值 | 货币、百分比、千分位格式、会计负数 | `"$50,000"` → `50000.0`，`"(20000)"` → `-20000.0` |
| 日期时间 | 同一列中的混合日期格式 | `"2021-01-15"`、`"01/20/2021"`、`"March 5, 2022"` → `datetime64` |
| 布尔值 | Yes/No、True/False、Y/N、1/0 | `"Yes"`/`"No"` → `True`/`False` |
| 分类 | 低基数文本（≤50 个唯一值，≤50% 行数） | `"Sales"`、`"IT"`、`"HR"` → `category` 类型 |
| 文本 | 其余所有 | 保持为字符串 |

仅当某列 **≥80%** 的非空值解析成功时，才执行转换——避免破坏真正包含混合内容的列。

### 5. 缺失值处理

- **数值列** → 中位数（默认）、均值、或填零
- **日期列** → 线性插值 + 前向/后向填充
- **布尔列** → 众数（出现频率最高的值）
- **分类/文本列** → 众数，或标记为 `"Missing"`
- **缺失超过 60% 的列** → 直接删除

### 6. 异常值处理（IQR 方法）

针对数值列（不含 0/1 二值标记列）：

- **截尾（Winsorize）**——裁剪至 `[Q1 - k×IQR, Q3 + k×IQR]`（默认，k=1.5）
- **删除**——删除包含异常值的行
- **不处理**——仅检测并报告，不修改数据

### 7. 数据脱敏 🆕

通过列名和内容模式自动识别敏感列：

- **姓名** → `张三` → `张*`，`Alice Wang` → `A****`
- **手机号** → `13812345678` → `138****5678`
- **身份证号** → `110101199001011234` → `110101********1234`
- **邮箱** → `zhangsan@company.com` → `z****@company.com`
- **地址** → `北京市朝阳区中山路100号` → `北京市朝阳区******`

可在侧边栏单独开关每种脱敏类别。

### 8. 数据质量报告 🆕

- **质量评分**（0-100）：综合完整度（50%）、唯一性（20%）、类型一致性（30%）计算得出
- 清洗前后完整度对比
- 数值列统计信息（均值、标准差、四分位数、最小值/最大值）

### 9. 数据可视化 🆕

- **缺失值柱状图**——清洗前后对比
- **异常值箱线图**——每个受影响数值列的处理前后并排对比
- **分布直方图**——叠加展示异常值处理前后的分布变化

### 10. Excel 支持 🆕

- 导入：`.csv`、`.xlsx`、`.xls`
- 导出：CSV 和 Excel 双格式下载按钮

---

## 输出内容

- 清洗前后并排预览
- 标签页式报告：质量评分、可视化图表、类型转换记录、缺失值处理明细、异常值边界、脱敏摘要、列名变更
- **下载清洗后 CSV** 和 **下载清洗后 Excel** 按钮
- 完整 JSON 清洗报告（可展开查看）

---

## 运行测试

```bash
pip install pytest
pytest tests/ -v
```

测试覆盖 `cleaning_engine.py` 中所有核心功能，包括：

- 列名标准化（特殊字符、重复列名、空列名）
- 重复行删除
- 类型检测（货币、日期、布尔值、分类、混合内容）
- 缺失值填补（全部策略）
- 异常值处理（截尾/删除/不处理，IQR 灵敏度）
- 敏感列检测与脱敏
- 完整流水线集成（含进度回调）

---

## 生成测试数据

```bash
# 1 万行（默认）
python generate_sample_data.py

# 100 万行
python generate_sample_data.py --rows 1000000 --output million_rows.csv

# 自定义噪声水平
python generate_sample_data.py --rows 50000 --noise 0.15 --seed 123
```

参数说明：

- `--rows`——生成的大致行数（默认：10000）
- `--output`——输出的 CSV 文件路径
- `--seed`——随机种子，保证结果可复现
- `--noise`——刻意引入问题的行比例（默认：0.1）

生成字段：员工姓名、薪资、入职日期、是否在职、部门、年龄、绩效评级、手机号、邮箱、身份证号、地址——包含故意的格式混乱、缺失值、异常值和重复行。

---

## 项目结构

```
AI训练数据智能清洗平台/
├── app.py                      # Streamlit 界面
├── cleaning_engine.py          # 纯逻辑清洗流水线（不依赖 Streamlit）
├── generate_sample_data.py     # 脏数据生成器（用于测试）
├── requirements.txt
├── sample_messy_data.csv       # 小规模示例文件
├── README.md
└── tests/
    ├── __init__.py
    └── test_cleaning_engine.py # pytest 单元测试
```

`cleaning_engine.py` 也可单独导入使用：

```python
import pandas as pd
from cleaning_engine import clean_dataframe

df = pd.read_csv("my_data.csv")
cleaned_df, report = clean_dataframe(
    df,
    enable_masking=True,  # 启用敏感数据脱敏
    progress_callback=lambda i, t, n: print(f"[{i+1}/{t}] {n}"),
)
```

---

## 技术栈

- **Python 3.10+**
- **Streamlit**——Web 界面
- **Pandas**——数据处理
- **NumPy**——数值计算
- **Matplotlib**——数据可视化
- **OpenPyXL**——Excel 文件支持
- **pytest**——单元测试
