import pandas as pd
import numpy as np

# Setting display options to show all columns
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1500)  # Increase display width if needed

row_limit = 200

# Read Excel sheets with row limit
df_bat = pd.read_excel(r"D:\Cricket Databases\CC_Database_v6.1.xlsx", sheet_name='Bat', nrows=row_limit)
df_bowl = pd.read_excel(r"D:\Cricket Databases\CC_Database_v6.1.xlsx", sheet_name='Bowl', nrows=row_limit)
df_gs = pd.read_excel(r"D:\Cricket Databases\CC_Database_v6.1.xlsx", sheet_name='GS', nrows=row_limit)
df_match = pd.read_excel(r"D:\Cricket Databases\CC_Database_v6.1.xlsx", sheet_name='Matches', nrows=row_limit)
df_pp = pd.read_excel(r"D:\Cricket Databases\CC_Database_v6.1.xlsx", sheet_name='PP', nrows=row_limit)
df_tables = pd.read_excel(r"D:\Cricket Databases\CC_Database_v6.1.xlsx", sheet_name='Season Tables', nrows=row_limit)
df_bbb = pd.read_excel(r"D:\Cricket Databases\CC_Database_v6.1.xlsx", sheet_name='BBB', nrows=row_limit)

df_bat.head(50)
