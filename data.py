import pandas as pd
from datetime import datetime


def load_data(path):
    return pd.read_csv(path)


def add_new_columns(df):
    add_season_name(df)
    df['datetime'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
    add_hour(df)
    add_day(df)
    add_month(df)
    add_year(df)
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
    df['is_weekend_holiday'] = df.apply(lambda x: weekend_orand_holiday(x.is_holiday, x.is_weekend), axis=1)


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
    print()
    get_top5_bot5(corr)
    print()
    means_by_t_diff(df)
    print()
    # means_array = means_by_t_diff(df)
    # print_corr_t_diff(means_array)


# data_analysis aiding methods
def get_top5_bot5(df):
    """
    creates a correlation dictionary from the df, then gets the top 5 (high to low) and bottom 5 (low to high)
    :param df: the dataframe were turning into a dict
    :return: prints the top 5 and bottom 5
    """
    corr_dictionary = dict()
    for i, row in zip(range(len(df.index)), df.index):
        for j, column in enumerate(df.columns[i + 1:len(df.columns)]):
            key = (row, column)
            corr_dictionary[key] = abs(df.get(row)[column])

    sorted_list = sorted(corr_dictionary.items(), key=lambda x: x[1], reverse=True)
    sorted_list_reversed = sorted(corr_dictionary.items(), key=lambda x: x[1])
    top_5 = sorted_list[:5]
    bot_5 = sorted_list_reversed[:5]
    print("Highest correlated are: ")
    for i, item in enumerate(top_5):
        print(f"{i + 1}. {item[0]} with {item[1]:.6f}")
    print()
    print("Lowest correlated are: ")
    for i, item in enumerate(bot_5):
        print(f"{i + 1}. {item[0]} with {item[1]:.6f}")


def print_corr_details(dictionary):
    for i, key, val in enumerate(dictionary):
        print(f"{i}. {key} with {val:.6f}")


def means_by_t_diff(df):
    df_groupby = df.groupby(['season_name'])['t_diff'].mean().to_frame()
    for season, mean in zip(df_groupby.index, df_groupby['t_diff']):
        print(f"{season} average t_diff is {mean:.2f}")
    print(f"All average t_diff is {df['t_diff'].mean():.2f}")

    # df_groupby = df.groupby(['season_name'])['t_diff'].mean().to_frame(name='mean_t_diff').reset_index()
    # total_means.append((df_groupby.iloc[i, 0], df_groupby.iloc[i, 1]) for i in range(4))
    # total_means.append(('All', df['t_diff'].mean()))
    # return total_means


def print_corr_t_diff(means):
    for t_mean in means:
        print(f"{t_mean[0]:.2f} average t_diff is {t_mean[1]:.2f}")
