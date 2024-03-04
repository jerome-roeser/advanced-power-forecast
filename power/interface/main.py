
from power.ml_ops.data import get_pv_data, clean_pv_data


###

df = get_pv_data()
df = clean_pv_data(df)
df.head()
