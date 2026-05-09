import pandas as pd
import numpy as np

df = pd.read_excel(r'D:\data\thesis_5min_edited2.xlsx', sheet_name='5-min Data')
df['window_end'] = pd.to_datetime(df['window_end (UTC)'])
df = df.sort_values('window_end').reset_index(drop=True)

# Compute log return in basis points
df['return'] = 10000 * np.log(df['dex_pool_price'] / df['dex_pool_price'].shift(1))

# Identify gaps — where consecutive timestamps are more than 5 minutes apart
df['gap'] = (df['window_end'] - df['window_end'].shift(1)) > pd.Timedelta('5min')

# Blank out returns immediately following a gap
df.loc[df['gap'], 'return'] = np.nan

# Also blank out the first return (initialization)
df.loc[0, 'return'] = np.nan