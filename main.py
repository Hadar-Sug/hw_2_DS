import clustering
import data
import sys
import numpy as np


def part_a(argv):
    df = data.load_data(argv[1])
    # data.add_new_columns(df)
    # data.data_analysis(df)
    df_trans = clustering.transform_data(df, ['cnt', 'hum'])
    for k in [2, 3, 5]:
        print (f'k = {k}')
        labels, centroids = clustering.kmeans(df_trans, k)
        print(np.array_str(centroids, precision=3, suppress_small=True))
        print()




def main(argv):
    # print("Part A: ")
    part_a(argv)


if __name__ == '__main__':
    main(sys.argv)
