import time
from datetime import datetime
import os

import multiLayerNeuralNetwork


def __handle_file__(contents: list):
    lines = []
    for line in contents[1:]:
        split = line.split(',')
        date, open_, high, low, close, volume, open_int = split
        date = datetime.strptime(date, '%Y-%m-%d')
        timestamp = int(time.mktime(date.timetuple()))
        data_point = DataPoint(timestamp, float(open_), float(high), float(low), float(close), float(volume), int(open_int))
        lines.append(data_point)
    return lines


class Data:
    def __init__(self, root_path: str):
        self.etf = []
        self.stock = []
        __read_data__(root_path + "/ETFs", self.etf)
        __read_data__(root_path + "/Stocks", self.stock)


class DataPoint:
    def __init__(self, date: int, open_: float, high: float, low: float, close: float, volume: float,
                 open_int: int):
        self.date = date
        self.open_ = open_
        self.close = close
        self.volume = volume


def __read_data__(root_path: str, insertion_list: list):
    files = os.listdir(root_path)
    for file in files:
        file_path = os.path.join(root_path, file)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                contents = f.readlines()
                data_points = __handle_file__(contents)
                insertion_list.extend(data_points)


def main():
    data = Data("samples/stock_market")
    ann = multiLayerNeuralNetwork.MultiLayerNeuralNetwork(3, [15, 10, 5], 2, 0.25)
    for data_point in data.stock:
        print(data_point)
        ann.train([data_point.date, data_point.open_, data_point.volume], [data_point.close])


if __name__ == '__main__':
    main()
