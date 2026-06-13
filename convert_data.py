import pandas as pd
import os

excel_path = 'data/online_retail_II.xlsx'
csv_path = 'data/online_retail.csv'

if os.path.exists(excel_path):
    print("Reading Excel file...")
    # The dataset has two sheets: 'Year 2009-2010' and 'Year 2010-2011'
    df1 = pd.read_excel(excel_path, sheet_name='Year 2009-2010')
    df2 = pd.read_excel(excel_path, sheet_name='Year 2010-2011')
    df = pd.concat([df1, df2])
    print(f"Combined shape: {df.shape}")
    df.to_csv(csv_path, index=False, encoding='latin-1')
    print(f"Saved to {csv_path}")
else:
    print(f"File {excel_path} not found.")
