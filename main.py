import clustering
import data
import sys
import numpy as np


def part_a(the_data):
    data.add_new_columns(the_data)
    data.data_analysis(the_data)


def part_b(the_data):
    df_trans = clustering.transform_data(the_data, ['cnt', 'hum'])
    for k in [2, 3, 5]:
        print(f'k = {k}')
        labels, centroids = clustering.kmeans(df_trans, k)
        print(np.array_str(centroids, precision=3, suppress_small=True))
        print()
        clustering.visualize_results(df_trans, labels, centroids, "/home/student/projects/hw2/")


def main():
    path = "./london_sample_500.csv"
    df = data.load_data(path)
    print("Part A: ")
    part_a(df)
    print("Part B: ")
    part_b(df)


if __name__ == '__main__':
    main()
