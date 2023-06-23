import os
import random
import time

import numpy

import helper
import multiLayerNeuralNetwork


def __handle_file__(contents: list):
    lines = []
    for line in contents[1:]:
        split = line.split(',')
        date, open_, high, low, close, volume, open_int = split
        timestamp = helper.convert_to_unix_timestamp(date)
        data_point = DataPoint(timestamp, float(open_), float(high), float(low), float(close), float(volume),
                               int(open_int))
        lines.append(data_point)
    return lines


def __normalize_data__(data_points: list):
    max_date = max_open = max_close = max_volume = float('-inf')
    for dp in data_points:
        if dp.date > max_date:
            max_date = dp.date
        if dp.open_ > max_open:
            max_open = dp.open_
        if dp.close > max_close:
            max_close = dp.close
        if dp.volume > max_volume:
            max_volume = dp.volume
    return max_date, max_open, max_close, max_volume


class Data:
    def __init__(self, root_path: str, etf_percentage: float, stock_percentage: float):
        self.etf = __read_data__(root_path + "/ETFs", etf_percentage)
        self.stock = __read_data__(root_path + "/Stocks", stock_percentage)


class DataPoint:
    def __init__(self, date: int, open_: float, high: float, low: float, close: float, volume: float,
                 open_int: int):
        self.date = date
        self.open_ = open_
        self.close = close
        self.volume = volume

    def as_list(self):
        return [self.date, self.open_, self.volume]


def __read_data__(root_path, percentage: float):
    files = os.listdir(root_path)
    num_files_to_read = int(len(files) * percentage)
    files_to_read = random.sample(files, num_files_to_read)
    insertion_list = []
    for file in files_to_read:
        file_path = os.path.join(root_path, file)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                contents = f.readlines()
                data_points = __handle_file__(contents)
                insertion_list.extend(data_points)
    return insertion_list


def main():
    epochs = 2
    data = Data("samples/stock_market", 0.0, 0.5)
    ann = multiLayerNeuralNetwork.MultiLayerNeuralNetwork(3, [15, 10, 5], 3, 0.25)
    print(f"Starting training with {len(data.stock)} data points")
    start_time = time.time()
    for i in range(epochs):
        print(f"Training epoch {i + 1}...")
        for data_point in data.stock:
            # print(f"training with {data_point.date}, {data_point.open_}, {data_point.volume}, expecting {
            # data_point.close}")
            ann.train(data_point.as_list(), [__return_change_(data_point.open_, data_point.close)])
    training_time = time.time() - start_time
    formatted_runtime = helper.format_runtime(training_time)
    print(f"Training took: {formatted_runtime}h")
    performance = int(test_network(ann, data) * 100)
    print(f"Network reached {performance}% accuracy")


def test_network(ann: multiLayerNeuralNetwork.MultiLayerNeuralNetwork, data: Data):
    scorecard = []
    for data_point in data.stock:
        result = ann.query([data_point.date, data_point.open_, data_point.volume])
        # 0 means dropped value, 1 means no significant change, 2 means improved
        index = numpy.argmax(result)
        score = __handle_comparison(index, data_point.open_, data_point.close)
        scorecard.append(score)
    scorecard_array = numpy.asarray(scorecard)
    score = scorecard_array.sum() / scorecard_array.size
    # print("performance = ", score)
    return score


def __handle_comparison(actual_value: int, open_: float, close: float):
    expected = __return_change_(open_, close)
    if actual_value == expected:  # return score
        return 1
    else:
        return 0


def __return_change_(open_: float, close: float):
    if open_ < close:
        expected = 2
    elif open_ > close:
        expected = 0
    else:
        expected = 1
    return expected


if __name__ == '__main__':
    main()
