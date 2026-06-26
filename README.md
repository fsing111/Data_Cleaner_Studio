# 🧹 Automated Data Cleaning Studio

A Streamlit web app that takes a messy CSV or Excel file and returns a cleaned,
properly-typed, download-ready version — with a full transparency report,
data quality scoring, visualizations, and optional data masking.

---

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload `sample_messy_data.csv` to see
a worked example.

For large-scale testing, generate synthetic dirty data:

```bash
python generate_sample_data.py --rows 100000 --output big_data.csv
```

---

## What it does

### 1. Column name standardization
`  Employee Name  ` → `employee_name`
`Salary ($)` → `salary`
`Is Active?` → `is_active`

Lowercase, snake_case, special characters stripped, whitespace collapsed,
duplicate names auto-numbered.

### 2. Whitespace & text normalization
Trims leading/trailing spaces and collapses internal multi-spaces
in every text column (`"  good  "` → `"good"`).

### 3. Duplicate removal
Drops exact duplicate rows.

### 4. Smart type detection & conversion
For every text/object column, tries — in order:

| Step | Detects | Example |
|---|---|---|
| Numeric | Currency, percent, comma-formatted numbers, accounting negatives | `"$50,000"` → `50000.0`, `"(20000)"` → `-20000.0` |
| Datetime | Mixed date formats in the same column | `"2021-01-15"`, `"01/20/2021"`, `"March 5, 2022"` → `datetime64` |
| Boolean | Yes/No, True/False, Y/N, 1/0 | `"Yes"`/`"No"` → `True`/`False` |
| Category | Low-cardinality text (≤50 unique, ≤50% of rows) | `"Sales"`, `"IT"`, `"HR"` → `category` dtype |
| Text | Everything else | kept as string |

A column converts only if **≥80%** of its non-null values parse successfully —
this avoids corrupting genuinely mixed-content columns.

### 5. Missing value handling
- **Numeric** → median (default), mean, or zero
- **Datetime** → linear interpolation + forward/back fill
- **Boolean** → mode (most frequent value)
- **Categorical/text** → mode, or literal `"Missing"` label
- **Columns >60% missing** → dropped entirely

### 6. Outlier handling (IQR method)
For numeric columns (excluding binary 0/1 flags):
- **Cap (winsorize)** — clip to `[Q1 - k×IQR, Q3 + k×IQR]` (default, k=1.5)
- **Remove** — drop rows containing outliers
- **None** — detect only, report without modifying

### 7. Data Masking 🆕
Auto-detects sensitive columns by column name and content patterns:
- **Names** → `张三` → `张*`, `Alice Wang` → `A****`
- **Phone numbers** → `13812345678` → `138****5678`
- **ID cards** → `110101199001011234` → `110101********1234`
- **Emails** → `zhangsan@company.com` → `z****@company.com`
- **Addresses** → `北京市朝阳区中山路100号` → `北京市朝阳区******`

Toggle individual masking categories on/off from the sidebar.

### 8. Data Quality Report 🆕
- **Quality Score** (0-100): composite score based on completeness (50%), uniqueness (20%), and type consistency (30%)
- Before/after completeness comparison
- Numeric column statistics (mean, std, quartiles, min/max)

### 9. Data Visualizations 🆕
- **Missing values bar chart** — before vs after comparison
- **Outlier box plots** — side-by-side before/after for each affected numeric column
- **Distribution histograms** — overlaid before/after histograms showing how outlier treatment reshaped distributions

### 10. Excel Support 🆕
- Import: `.csv`, `.xlsx`, `.xls`
- Export: both CSV and Excel download buttons

---

## Output

- Side-by-side before/after preview
- Tabbed report: quality score, visualizations, type conversions, missing-value actions, outlier bounds, masking summary, column renames
- **Download Cleaned CSV** and **Download Cleaned Excel** buttons
- Full JSON cleaning report (expandable)

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests cover all core `cleaning_engine.py` functions including:
- Column name standardization (special chars, duplicates, empty names)
- Duplicate removal
- Type detection (currency, dates, booleans, categories, mixed content)
- Missing value imputation (all strategies)
- Outlier handling (cap/remove/none, IQR sensitivity)
- Sensitive column detection and masking
- Full pipeline integration with progress callback

---

## Generating Test Data

```bash
# 10K rows (default)
python generate_sample_data.py

# 1 million rows
python generate_sample_data.py --rows 1000000 --output million_rows.csv

# Custom noise level
python generate_sample_data.py --rows 50000 --noise 0.15 --seed 123
```

Options:
- `--rows` — approximate number of rows (default: 10000)
- `--output` — output CSV path
- `--seed` — random seed for reproducibility
- `--noise` — fraction of rows with intentional issues (default: 0.1)

Generated columns: Employee Name, Salary, Join Date, Is Active, Department, Age, Performance Rating, Phone, Email, ID Card, Address — with intentionally messy formatting, missing values, outliers, and duplicates.

---

## Files

```
AI训练数据智能清洗平台/
├── app.py                      # Streamlit UI
├── cleaning_engine.py          # Pure-logic cleaning pipeline (no Streamlit deps)
├── generate_sample_data.py     # Dirty data generator for testing
├── requirements.txt
├── sample_messy_data.csv       # Small test file
├── README.md
└── tests/
    ├── __init__.py
    └── test_cleaning_engine.py # pytest unit tests
```

`cleaning_engine.py` can also be imported and used standalone:

```python
import pandas as pd
from cleaning_engine import clean_dataframe

df = pd.read_csv("my_data.csv")
cleaned_df, report = clean_dataframe(
    df,
    enable_masking=True,  # mask sensitive data
    progress_callback=lambda i, t, n: print(f"[{i+1}/{t}] {n}"),
)
```

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — web UI
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib** — data visualizations
- **OpenPyXL** — Excel file support
- **pytest** — testing
