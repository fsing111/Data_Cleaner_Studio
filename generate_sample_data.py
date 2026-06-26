"""
Dirty Sample Data Generator
============================
Generates a CSV file with intentionally messy data for testing the
data cleaning platform. Configurable row count, noise level, and
output path.

Usage:
    python generate_sample_data.py                          # 10K rows → sample_messy_data_large.csv
    python generate_sample_data.py --rows 1000000           # 1M rows
    python generate_sample_data.py --rows 50000 --output my_data.csv --seed 42
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────
# Chinese surnames and given names for realistic name generation
SURNAMES = [
    "王", "李", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴",
    "徐", "孙", "胡", "朱", "高", "林", "何", "郭", "马", "罗",
    "梁", "宋", "郑", "谢", "韩", "唐", "冯", "于", "董", "萧",
]

GIVEN_NAMES_MALE = [
    "伟", "强", "磊", "洋", "勇", "军", "杰", "涛", "明", "超",
    "建华", "志强", "文博", "浩然", "子涵", "宇轩", "梓豪", "一鸣",
]

GIVEN_NAMES_FEMALE = [
    "芳", "敏", "静", "丽", "婷", "雪", "玲", "萍", "红", "霞",
    "秀英", "美玲", "雅琪", "诗涵", "雨彤", "思雨", "心怡", "若曦",
]

ALL_GIVEN = GIVEN_NAMES_MALE + GIVEN_NAMES_FEMALE

DEPARTMENTS = ["Sales", "IT", "HR", "Marketing", "Finance", "Operations", "R&D", "Legal"]
DEPT_VARIANTS = {
    "Sales": ["sales", "SALES", "Sales ", " sales", "Sale"],
    "IT": ["it", "IT ", "I.T.", "Information Technology"],
    "HR": ["hr", "HR ", "H.R.", "Human Resources"],
    "Marketing": ["marketing", "MARKETING", "Marketing ", "Mktg"],
    "Finance": ["finance", "Finance ", "FIN", "Financial"],
    "Operations": ["ops", "Ops", "Operations ", "OPERATIONS"],
    "R&D": ["R&D", "r&d", "RnD", "R and D"],
    "Legal": ["legal", "Legal ", "LEGAL"],
}

CITIES = [
    "北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京",
    "西安", "重庆", "长沙", "青岛", "大连", "厦门", "苏州", "天津",
]


def generate_phone(rng: np.random.Generator) -> str:
    """Generate a plausible Chinese mobile number."""
    prefixes = ["130", "131", "132", "133", "134", "135", "136", "137", "138", "139",
                "150", "151", "152", "153", "155", "156", "157", "158", "159",
                "180", "181", "182", "183", "184", "185", "186", "187", "188", "189"]
    prefix = rng.choice(prefixes)
    suffix = "".join(str(rng.integers(0, 10)) for _ in range(8))
    return prefix + suffix


def generate_id_card(rng: np.random.Generator) -> str:
    """Generate a plausible 18-digit Chinese ID card number."""
    area = str(rng.integers(110000, 659000))
    birth = f"{rng.integers(1960, 2010):04d}{rng.integers(1, 13):02d}{rng.integers(1, 29):02d}"
    suffix = "".join(str(rng.integers(0, 10)) for _ in range(3))
    checksum = rng.choice(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "X"])
    return area + birth + suffix + checksum


def generate_email(name: str, rng: np.random.Generator) -> str:
    """Generate a plausible email from a name."""
    domains = ["company.com", "email.com", "corp.cn", "mail.cn", "example.org"]
    # Pinyin-ish simplification: just use the raw name or initials
    if rng.random() < 0.5:
        local = name.lower().replace(" ", ".") + str(rng.integers(0, 999))
    else:
        local = name[:1].lower() + name[1:].lower() + str(rng.integers(0, 99))
    return local + "@" + rng.choice(domains)


def format_salary(salary: float, rng: np.random.Generator) -> str:
    """Format salary with various messy formats."""
    fmt = rng.choice(["clean", "dollar", "comma", "accounting", "euro"])
    if fmt == "clean":
        return str(int(salary))
    elif fmt == "dollar":
        return f"${salary:,.0f}"
    elif fmt == "comma":
        return f"{salary:,.0f}"
    elif fmt == "accounting":
        # (50000) style negative — but salary is positive so use ( ) rarely
        return f"({salary:,.0f})" if rng.random() < 0.05 else f"{salary:,.0f}"
    else:
        return f"€{salary:,.0f}"


def format_date(date, rng: np.random.Generator) -> str:
    """Format a date with various messy formats."""
    fmt = rng.choice([
        "%Y-%m-%d",           # 2021-01-15
        "%m/%d/%Y",           # 01/20/2021
        "%d/%m/%Y",           # 20/01/2021
        "%B %d, %Y",          # March 5, 2022
        "%b %d, %Y",          # Mar 5, 2022
        "%Y%m%d",             # 20210115
        "%d-%b-%Y",           # 15-Jan-2021
    ])
    return date.strftime(fmt)


def format_boolean(val: bool, rng: np.random.Generator) -> str:
    """Format boolean with various styles."""
    fmt = rng.choice(["yesno", "truefalse", "yn", "10", "upper", "lower", "mixed"])
    if val:
        options = {"yesno": "Yes", "truefalse": "True", "yn": "Y", "10": "1",
                   "upper": "TRUE", "lower": "true", "mixed": "Yes"}
    else:
        options = {"yesno": "No", "truefalse": "False", "yn": "N", "10": "0",
                   "upper": "FALSE", "lower": "false", "mixed": "No"}
    return options[fmt]


def generate_dataframe(n_rows: int, seed: int = 42, noise_level: float = 0.1) -> pd.DataFrame:
    """
    Generate a messy DataFrame.

    Parameters
    ----------
    n_rows : int
        Number of rows to generate (will be ~5-10% larger before dedup)
    noise_level : float
        Fraction of rows with intentional issues (missing, outliers, etc.)
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Generate slightly more rows to account for intentional duplicates
    n_gen = int(n_rows * 1.05)
    noise_n = max(1, int(n_gen * noise_level))

    data = {
        "Employee Name": [],
        "Salary ($)": [],
        "Join Date": [],
        "Is Active?": [],
        "Department": [],
        "Age": [],
        "Performance Rating": [],
        "Phone": [],
        "Email": [],
        "ID Card": [],
        "Address": [],
    }

    for i in range(n_gen):
        surname = rng.choice(SURNAMES)
        given = rng.choice(ALL_GIVEN)
        name = surname + given

        salary = round(rng.uniform(30000, 200000), 2)
        join_date = pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(rng.integers(0, 2000)))
        is_active = rng.random() < 0.8
        department = rng.choice(DEPARTMENTS)
        age = int(rng.integers(22, 60))
        perf_rating = round(rng.uniform(2.0, 5.0), 1)
        phone = generate_phone(rng)
        email = generate_email(name, rng)
        id_card = generate_id_card(rng)
        city = rng.choice(CITIES)
        address = f"{city}{rng.choice(['朝阳区','海淀区','浦东新区','天河区','西湖区'])}" \
                  f"{rng.choice(['中山路','人民路','建设路','解放路','文化路'])}{rng.integers(1, 500)}号"

        data["Employee Name"].append(name)
        data["Salary ($)"].append(format_salary(salary, rng))
        data["Join Date"].append(format_date(join_date, rng))
        data["Is Active?"].append(format_boolean(is_active, rng))
        data["Department"].append(department)
        data["Age"].append(age)
        data["Performance Rating"].append(perf_rating)
        data["Phone"].append(phone)
        data["Email"].append(email)
        data["ID Card"].append(id_card)
        data["Address"].append(address)

    df = pd.DataFrame(data)

    # ── Inject noise ──────────────────────────────────────────────────────
    noise_indices = rng.choice(df.index, size=noise_n, replace=False)

    for idx in noise_indices:
        issue_type = rng.choice([
            "missing_name", "missing_salary", "missing_date", "missing_dept",
            "missing_age", "missing_rating", "missing_phone", "missing_email",
            "outlier_age_high", "outlier_age_low", "outlier_rating",
            "messy_dept", "messy_date", "invalid_date", "bad_phone",
            "bad_email", "bad_id_card",
        ])

        if issue_type == "missing_name":
            df.at[idx, "Employee Name"] = np.nan
        elif issue_type == "missing_salary":
            df.at[idx, "Salary ($)"] = np.nan
        elif issue_type == "missing_date":
            df.at[idx, "Join Date"] = np.nan
        elif issue_type == "missing_dept":
            df.at[idx, "Department"] = np.nan
        elif issue_type == "missing_age":
            df.at[idx, "Age"] = np.nan
        elif issue_type == "missing_rating":
            df.at[idx, "Performance Rating"] = np.nan
        elif issue_type == "missing_phone":
            df.at[idx, "Phone"] = np.nan
        elif issue_type == "missing_email":
            df.at[idx, "Email"] = np.nan
        elif issue_type == "outlier_age_high":
            df.at[idx, "Age"] = int(rng.integers(150, 300))
        elif issue_type == "outlier_age_low":
            df.at[idx, "Age"] = int(rng.integers(-20, 0))
        elif issue_type == "outlier_rating":
            df.at[idx, "Performance Rating"] = round(rng.uniform(-5.0, 15.0), 1)
        elif issue_type == "messy_dept":
            orig_dept = df.at[idx, "Department"]
            if orig_dept in DEPT_VARIANTS:
                df.at[idx, "Department"] = rng.choice(DEPT_VARIANTS[orig_dept])
        elif issue_type == "messy_date":
            df.at[idx, "Join Date"] = f"  {df.at[idx, 'Join Date']}  "
        elif issue_type == "invalid_date":
            df.at[idx, "Join Date"] = "not-a-date"
        elif issue_type == "bad_phone":
            bad_fmt = rng.choice([
                "138-1234-5678", "138123456", "138123456789", "abc-1234",
                "(138)12345678", "+86-13812345678",
            ])
            df.at[idx, "Phone"] = bad_fmt
        elif issue_type == "bad_email":
            df.at[idx, "Email"] = rng.choice([
                "not-an-email", "missing-at-sign.com", "user@", "@domain.com",
            ])
        elif issue_type == "bad_id_card":
            df.at[idx, "ID Card"] = rng.choice([
                "123456789012345678", "abcdefghi", "11010119900101123", "",
            ])

    # ── Add exact duplicate rows ──────────────────────────────────────────
    n_dupes = max(1, int(n_gen * 0.03))
    dupe_indices = rng.choice(df.index, size=n_dupes, replace=False)
    dupes = df.loc[dupe_indices].copy()
    df = pd.concat([df, dupes], ignore_index=True)

    # ── Add rows with all-empty values ────────────────────────────────────
    empty_row = {col: np.nan for col in df.columns}
    df = pd.concat([df, pd.DataFrame([empty_row, empty_row])], ignore_index=True)

    # ── Trim to approximately n_rows (accounting for dupes and empties) ───
    if len(df) > n_rows:
        df = df.iloc[:n_rows]

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate a messy CSV for testing the Data Cleaning Studio."
    )
    parser.add_argument(
        "--rows", type=int, default=10_000,
        help="Approximate number of rows to generate (default: 10000)"
    )
    parser.add_argument(
        "--output", type=str, default="sample_messy_data_large.csv",
        help="Output CSV file path (default: sample_messy_data_large.csv)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--noise", type=float, default=0.1,
        help="Fraction of rows with intentional noise (default: 0.1)"
    )

    args = parser.parse_args()

    print(f"Generating ~{args.rows:,} rows of messy data (seed={args.seed}, noise={args.noise})...")
    df = generate_dataframe(n_rows=args.rows, seed=args.seed, noise_level=args.noise)

    output_path = Path(args.output)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Done! {len(df):,} rows × {len(df.columns)} columns → {output_path} ({file_size_mb:.1f} MB)")

    # Quick stats
    print(f"\n📊 Quick stats:")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Missing values: {df.isna().sum().sum():,} total")
    print(f"   Duplicate rows: {df.duplicated().sum():,}")
    print(f"   Department values: {sorted(df['Department'].dropna().unique())}")


if __name__ == "__main__":
    main()
