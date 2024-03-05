
from power.ml_ops.data import get_pv_data, clean_pv_data
from power.ml_ops.model import model_yesterday


###

def dummy():
    df = get_pv_data()
    df = clean_pv_data(df)
    print(df.head(2))
    pass


def create_yesterday_baseline(date, power_source='PV'):
    """
    create a simple baseline model based on the data form d-1.
    """

    pv_data = get_pv_data()
    pv_data_clean = clean_pv_data(pv_data)
    
    yesterday_baseline = model_yesterday(pv_data_clean, date)
    return yesterday_baseline

def create_mean_baseline(date, power_source='PV'):
    """
    create a simple baseline model based on the data form d-1.
    """

    pv_data = get_pv_data()
    pv_data_clean = clean_pv_data(pv_data)

    # <YOUR CODE>

    pass



if __name__ == '__main__':
    create_yesterday_baseline(date='1982-07-06 05:52:00', power_source='PV')
    create_mean_baseline(date='1982-07-06 05:52:00', power_source='PV')
