
from power.ml_ops.data import get_pv_data, clean_pv_data


###

def dummy():
    df = get_pv_data()
    df = clean_pv_data(df)
    print(df.head(2))
    pass


if __name__ == '__main__':
    dummy()

