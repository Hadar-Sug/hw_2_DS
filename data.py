import pandas as pd
from datetime import datetime


def load_data(path):
    return pd.read_csv(path)


def add_new_columns(df):
    add_season_name(df)
    df['datetime'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
    add_day(df)
    add_month(df)
    add_year(df)
    add_hour(df)
    add_is_weekend_holiday(df)
    add_t_diff(df)
    return df


# aiding methods for add_new_columns


def add_season_name(df):
    """
    adds column with season name as string
    :param df: the dataframe
    :return:
    """
    df['season_name'] = df['season'].apply(to_string_season)


def add_hour(df):
    """
    adds column with corresponding hour
    :param df: the dataframe
    :return:
    """
    df['Hour'] = df['datetime'].apply(lambda x: x.hour)


def add_day(df):
    """
    adds column with corresponding day
    :param df: the dataframe
    :return:
    """

    df['Day'] = df['datetime'].apply(lambda x: x.day)


def add_month(df):
    """
    adds column with corresponding month
    :param df: the dataframe
    :return:
    """
    df['Month'] = df['datetime'].apply(lambda x: x.month)


def add_year(df):
    """
    adds column with corresponding year
    :param df: the dataframe
    :return:
    """
    df['Year'] = df['datetime'].apply(lambda x: x.year)


def to_string_season(number):
    """
    adds a column with string type names based on integer column named season
    :param number: number represents a season
    :return: the corresponding season string
    """
    if number == 0:
        return "spring"
    elif number == 1:
        return "summer"
    elif number == 2:
        return "fall"
    else:
        return "winter"


# need to check if this implementation is in fact correct
def add_is_weekend_holiday(df):
    """
    adds a binary column which is true only if is_holiday and is_weekend are true
    :param df: the dataframe
    :return:
    """
    df['is_weekend_holiday'] = df.apply(lambda x: weekend_orand_holiday(x.is_weekend, x.is_holiday), axis=1)


def weekend_orand_holiday(x, y):
    if x == 0 and y == 0:
        return 0
    if x == 0 and y == 1:
        return 1
    if x == 1 and y == 0:
        return 2
    if x == 1 and y == 1:
        return 3


def add_t_diff(df):
    """
    adds column with the delta between t2 and t1 : t2-t1
    :param df: the dateframe
    :return:
    """
    df['t_diff'] = df.apply(lambda x: x.t2 - x.t1, axis=1)


# end of methods for add_new_columns


def data_analysis(df):
    print("describe output:")
    print(df.describe().to_string())
    print()
    print("corr output:")
    corr = df.corr()
    print(corr.to_string())
    print("test")
    top_5 = corr_dict(corr)
    # for item in top_5:
    #   print(item)

    # print_corr_details(low_5)
    # means_array = means_by_t_diff(df)
    # print_corr_t_diff(means_array)


# data_analysis aiding methods
def corr_dict(df):
    corr_dictionary = dict()
    for i, row in enumerate(df.index):
        for column in df.columns[i + 1:-1]:
            key = (row, column)
            corr_dictionary[key] = df.get(row)[column]
    sorted_list = sorted(corr_dictionary.items(), key=lambda x: x[1])
    print(sorted_list[0])
    return corr_dictionary

    # sorted_items = sorted(corr_dictionary.values(), key=lambda x: abs(x[1]))
    # corr_dictionary= list(corr_dictionary.items())
    # print(type(corr_dictionary))
    # corr_dictionary = corr_dictionary.sort(key=lambda y: abs(y[1]))

    # print()
    # print((sorted(corr_dictionary.items(), key=lambda x: abs(x[1]))))
    # top_5 = sorted_items[:5]
    # reversed_sorted_items = sorted(corr_dictionary.items(), key=lambda x: abs(x[1]), reverse=True)
    # bottom_5 = reversed_sorted_items[:5]


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
