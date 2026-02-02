import pandas as pd
from pathlib import Path

path = Path(r"e:/ai/help2.1/help2.1/数据验证数据/验证.xls")
print("Exists:", path.exists())
try:
    df = pd.read_excel(path)
    print(df.columns.tolist())
    print(df.head())
except Exception as exc:
    print("Error reading excel:", exc)
