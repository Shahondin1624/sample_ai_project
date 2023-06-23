import time
from datetime import datetime
import os
import re
from itertools import product

import numpy


def extract_parameters(name: str):
    extractor_pattern = r'\d+\.\d+|\d+'
    parameters = [float(match) for match in re.findall(extractor_pattern, name)]
    return parameters[0], parameters[1], parameters[2], parameters[3]


def format_runtime(duration: float):
    milliseconds = int((duration - int(duration)) * 1000)
    return time.strftime(f"%H:%M:%S.{milliseconds:03d}", time.gmtime(duration))


def generate_file_name(hidden_nodes: int, learning_rate: float, epochs: int, performance: float):
    return f'performance_{performance}_epochs_{epochs}_hidden_nodes_{hidden_nodes}_learning_rate_{learning_rate}.npz'


# returns all model_file_names that match the pattern
def get_all_models():
    pattern = r'^performance_\d+\.\d+_epochs_\d+_hidden_nodes_\d+_learning_rate_\d+\.\d+\.npz$'
    folder_path = "models"
    model_names = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and re.match(pattern, filename):
            model_names.append(filename)
    return model_names


def estimate_runtime_left(current_iteration: int, total_iterations: int, runtime: float):
    runtime_left = (total_iterations - current_iteration) * runtime
    formatted_runtime = format_runtime(runtime_left)
    return f'At current speed, the training will take at least another {formatted_runtime}h'


def get_model_by_parameters(hidden_nodes: int, learning_rate: float, epochs: int):
    pattern = rf'^performance_\d+\.\d+_epochs_{epochs}_hidden_nodes_{hidden_nodes}_learning_rate_{learning_rate}.npz$'
    for model in get_all_models():
        if re.match(pattern, model):
            return model
    return None


def create_parameter_permutations(epochs_lower: int, epochs_upper: int, hidden_nodes_lower: int,
                                  hidden_nodes_upper: int,
                                  hidden_nodes_step_rate: int, learning_rate_lower: int, learning_rate_upper: int,
                                  learning_rate_step_rate: int):
    permutations = product(range(epochs_lower, epochs_upper),
                           range(hidden_nodes_lower, hidden_nodes_upper, hidden_nodes_step_rate),
                           [lr / 100.0 for lr in
                            range(learning_rate_lower, learning_rate_upper, learning_rate_step_rate)])
    return permutations


def derive_file_path(name: str):
    file_path = "models/" + name
    return file_path


def extract_performance_from_model_name(name: str):
    return float(name.split("_")[1])


# loading and normalizing+preparing training/test data
def load_training_data(is_training: bool):
    if is_training:
        path = "samples/mnist_train.csv"
    else:
        path = "samples/mnist_test.csv"
    file = open(path, 'r')
    data = file.readlines()
    images = []
    labels = []
    for record in data:
        values = record.split(',')
        inputs = (numpy.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(10) + 0.01
        targets[int(values[0])] = 0.99
        images.append(inputs)
        labels.append(targets)
    file.close()
    return images, labels


def determine_best_performing_model():
    best_model, best_performance = None, 0.0
    for filename in get_all_models():
        current_name = filename
        current_performance = extract_performance_from_model_name(current_name)
        if current_performance > best_performance:
            best_performance = current_performance
            best_model = current_name
    return best_model, best_performance


def convert_to_unix_timestamp(date: str):
    date = datetime.strptime(date, '%Y-%m-%d')
    timestamp = int(time.mktime(date.timetuple()))
    return timestamp
