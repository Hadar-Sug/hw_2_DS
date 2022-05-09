import pandas as pd
from datetime import datetime


def load_data(path):
    return pd.read_csv(path)


def add_new_columns(df):
    add_season_name(df)
    add_day(df)
    add_month(df)
    add_year(df)
    add_hour(df)
    add_is_weekend_holiday(df)
    add_t_diff(df)
    return df


# aiding methods for add_new_columns


def add_season_name(df):
    df['season_name'] = df['season'].apply(to_string_season)


"""
adds column with corresponding hour 
"""


def add_hour(df):
    df['Hour'] = df['timestamp'].apply(lambda x: x.time.hour)


"""
adds column with corresponding day 
"""


def add_day(df):
    df['Day'] = df['timestamp'].apply(lambda x: x.date.day)


"""
adds column with corresponding month
"""


def add_month(df):
    df['Month'] = df['timestamp'].apply(lambda x: x.date.month)


"""
adds column with corresponding year
"""


def add_year(df):
    df['Year'] = df['timestamp'].apply(lambda x: x.date.year)


def to_string_season(number):
    """
    adds a column with string type names based on integer column named season
    """
    if number == 0:
        return "spring"
    elif number == 1:
        return "summer"
    elif number == 2:
        return "fall"
    else:
        return "winter"


"""
adds a binary column which is true only if is_holiday and is_weekend are true
"""


def add_is_weekend_holiday(df):
    df['is_weekend_holiday'] = df[['is_weekend', 'is_holiday']].apply(lambda x: x[0] and x[1])  # hope this works


"""
adds column with the delta between t2 and t1 : t2-t1
"""


def add_t_diff(df):
    df['t_diff'] = df[['t1', 't2']].apply(lambda x: x[1] - x[0])


# end of methods for add_new_columns


def data_analysis(df):
    print("describe output:")
    print(df.describe().to_string())
    print()
    print("corr output:")
    corr = df.corr()
    print(corr.to_string())
    print()
    correlation_dict, rev_correlation_dict = corr_dict(df)
    top_5 = correlation_dict[1:5]
    low_5 = reversed(rev_correlation_dict[1:5])
    print_corr_details(top_5)
    print()
    print_corr_details(low_5)
    means_array = means_by_t_diff(df)
    print_corr_t_diff(means_array)


# data_analysis aiding methods
def corr_dict(df):
    corr_dictionary = {('i', 'j'): df.corr(i, j) for i, j in df.columns if j != i}
    corr_dictionary = dict(sorted(corr_dictionary.items(), key=lambda x: abs(x[1])))
    corr_dictionary_reversed = dict(sorted(corr_dictionary.items(), key=lambda x: abs(x[1]), reverse=True))
    return corr_dictionary, corr_dictionary_reversed


def print_corr_details(dictionary):
    for i, key, val in enumerate(dictionary):
        print(f"{i}. {key} with {val:.6f}")


def means_by_t_diff(df):
    total_means = []
    df_groupby = df.groupby(['season_name'])['t_diff'].mean().to_frame(name='mean_t_diff').reset_index()
    total_means.append((df_groupby.iloc[i, 0], df_groupby.iloc[i, 1]) for i in range(4))
    total_means.append(('All', df['t_diff'].mean()))
    return total_means


def print_corr_t_diff(means):
    for t_mean in means:
        print(f"{t_mean[0]:.2f} average t_diff is {t_mean[1]:.2f}")
