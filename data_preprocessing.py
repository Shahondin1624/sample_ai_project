import os
import numpy
from sklearn import preprocessing

import helper


def read_files(root_path: str, file_handler):
    files = os.listdir(root_path)
    insertion_list = []
    for file in files:
        file_path = os.path.join(root_path, file)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                contents = f.readlines()
                data_points = file_handler(contents)
                insertion_list.extend(data_points)
    return insertion_list


def normalize(data_points: list):
    as_array = numpy.array(data_points)
    normalized_arr = preprocessing.normalize(as_array)
    return normalized_arr


def handle_stock_market_file(lines: list):
    contents = []
    for line in lines[1:]:
        split = line.split(',')
        date, open_, high, low, close, volume, open_int = split
        timestamp = helper.convert_to_unix_timestamp(date)
        data_point = [timestamp, float(open_), float(high), float(low), float(close), float(volume),
                      int(open_int)]
        contents.append(data_point)
    return contents


def export_to_file(normalized_array: numpy.array, file_path: str):
    # savez_compressed accepts named parameters that are then stored in the file
    numpy.savez_compressed(file_path, data=normalized_array)


def main():
    data = read_files("samples/stock_market/Stocks", handle_stock_market_file)
    normalized = normalize(data)
    export_to_file(normalized, "samples/stock_market/stocks.npz")
    pass


if __name__ == '__main__':
    main()
