import neuralNetwork
import numpy as numpy
import scipy.special as scipy
import matplotlib.pyplot as matplot
import dill
import time
import os
import re


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


def train_network(epochs: int, input_nodes: int, hidden_nodes: int, output_nodes: int, learning_rate: float):
    ann = neuralNetwork.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    images, labels = load_training_data(True)
    start_time = time.time()
    for epoch in range(epochs):
        # print(f"Training for epoch {epoch + 1}...")
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            ann.train(image, label)
    training_time = time.time() - start_time
    formatted_runtime = format_runtime(training_time)
    print(f"Training took: {formatted_runtime}h")
    return ann, training_time


def format_runtime(duration: float):
    milliseconds = int((duration - int(duration)) * 1000)
    return time.strftime(f"%H:%M:%S.{milliseconds:03d}", time.gmtime(duration))


def test_network(ann: neuralNetwork):
    scorecard = []
    test_images, test_labels = load_training_data(False)
    for index in range(len(test_images)):
        query_result = ann.query(test_images[index])
        result = numpy.argmax(query_result)
        expected = numpy.argmax(test_labels[index])
        if result == expected:
            scorecard.append(1)
        else:
            scorecard.append(0)
    scorecard_array = numpy.asarray(scorecard)
    score = scorecard_array.sum() / scorecard_array.size
    # print("performance = ", score)
    return score


def export_model(ann: neuralNetwork, name: str):
    file_path = derive_file_path(name)
    with open(file_path, 'wb') as file:
        dill.dump(ann, file)


def import_model(name: str):
    file_path = derive_file_path(name)
    with open(file_path, 'rb') as file:
        ann = dill.load(file)
        return ann


def derive_file_path(name: str):
    file_path = "models/" + name + ".bin"
    return file_path


def generate_file_name(hidden_nodes: int, learning_rate: float, epochs: int, performance: float):
    return f'performance_{performance}_epochs_{epochs}_hidden_nodes_{hidden_nodes}_learning_rate_{learning_rate}.bin'


def determine_best_performing_model():
    pattern = r'^performance_\d+\.\d+_epochs_\d+_hidden_nodes_\d+_learning_rate_\d+\.\d+\.bin$'
    folder_path = "models"
    best_model, best_performance = None, 0.0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and re.match(pattern, filename):
            current_name = filename
            current_performance = extract_performance_from_model_name(current_name)
            if current_performance > best_performance:
                best_performance = current_performance
                best_model = current_name
    return best_model, best_performance


def extract_performance_from_model_name(name: str):
    return float(name.split("_")[1])


def estimate_runtime_left(current_iteration: int, total_iterations: int, runtime: float):
    runtime_left = (total_iterations - current_iteration) * runtime
    formatted_runtime = format_runtime(runtime_left)
    return f'At current speed, the training will take at least another {formatted_runtime}h'


def already_tested(epochs: int, hidden_nodes: int, learning_rate: float):
    return True


# Will return the following values of the best performing model in this order: performance, epochs, hidden_nodes,
# learning_rate. lower/upper represent the bounds in between models will be tested. lower < upper for this code to work
def determine_best_parameters(epochs_lower: int, epochs_upper: int, hidden_nodes_lower: int, hidden_nodes_upper: int,
                              hidden_nodes_step_rate: int, learning_rate_lower: int, learning_rate_upper: int,
                              learning_rate_step_rate: int):
    input_nodes: int = 784
    output_nodes: int = 10

    total_iterations = int((epochs_upper - epochs_lower) * (
            (hidden_nodes_upper - hidden_nodes_lower) / hidden_nodes_step_rate) *
                           ((learning_rate_upper - learning_rate_lower) / learning_rate_step_rate))
    current_iteration = 1

    for epochs in range(epochs_lower, epochs_upper):
        for hidden_nodes in range(hidden_nodes_lower, hidden_nodes_upper, hidden_nodes_step_rate):
            for learning_rate_as_int in range(learning_rate_lower, learning_rate_upper,
                                              learning_rate_step_rate):
                learning_rate = learning_rate_as_int / 100.0
                if already_tested(epochs, hidden_nodes, learning_rate):
                    print(f"Already created a network with epochs={epochs}, "
                          f"hidden_nodes={hidden_nodes}, learning_rate={learning_rate}")
                    continue
                ann, training_time = train_network(epochs, input_nodes, hidden_nodes, output_nodes, learning_rate)
                performance = test_network(ann)
                name = generate_file_name(hidden_nodes, learning_rate, epochs, performance)
                print(
                    f"{name} - {current_iteration}/{total_iterations} - {estimate_runtime_left(current_iteration, total_iterations, training_time)}")
                current_iteration += 1
                export_model(ann, name)
    best_model, best_performance = determine_best_performing_model()
    print(f"Best performing model: {best_model} with a performance of {best_performance}")
    extractor_pattern = r'\d+\.\d+|\d+'
    parameters = [float(match) for match in re.findall(extractor_pattern, best_model)]
    return parameters[0], parameters[1], parameters[2], parameters[3]


def main():
    epochs_lower = 1
    epochs_upper = 11
    hidden_nodes_lower = 100
    hidden_nodes_upper = 260
    hidden_nodes_step_rate = 10
    learning_rate_lower = 10
    learning_rate_upper = 30
    learning_rate_step_rate = 1
    determine_best_parameters(epochs_lower, epochs_upper, hidden_nodes_lower, hidden_nodes_upper,
                              hidden_nodes_step_rate, learning_rate_lower, learning_rate_upper, learning_rate_step_rate)


if __name__ == '__main__':
    main()
