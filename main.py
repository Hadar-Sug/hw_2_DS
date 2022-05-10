import data
import sys


def part_a(argv):
    df = data.load_data(argv[1])
    data.add_new_columns(df)
    data.data_analysis(df)


def main(argv):
    print("part A")
    part_a(argv)


if __name__ == '__main__':
    main(sys.argv)
