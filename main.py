import time

import numpy as numpy

import helper
import neuralNetwork
from helper import extract_parameters, format_runtime, get_all_models, generate_file_name, estimate_runtime_left
from simpleCache import SimpleCache
from multiprocessing import Pool, Value, Process, Manager
from itertools import product


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


def determine_best_performing_model():
    best_model, best_performance = None, 0.0
    for filename in get_all_models():
        current_name = filename
        current_performance = extract_performance_from_model_name(current_name)
        if current_performance > best_performance:
            best_performance = current_performance
            best_model = current_name
    return best_model, best_performance


def extract_performance_from_model_name(name: str):
    return float(name.split("_")[1])


def already_tested(cache: SimpleCache, epochs: int, hidden_nodes: int, learning_rate: float):
    return cache.contains_tuple(hidden_nodes, epochs, learning_rate)


# Will return the following values of the best performing model in this order: performance, epochs, hidden_nodes,
# learning_rate. lower/upper represent the bounds in between models will be tested. lower < upper for this code to work
def determine_best_parameters(epochs_lower: int, epochs_upper: int, hidden_nodes_lower: int, hidden_nodes_upper: int,
                              hidden_nodes_step_rate: int, learning_rate_lower: int, learning_rate_upper: int,
                              learning_rate_step_rate: int, cache: SimpleCache, manager: Manager):
    total_iterations = int((epochs_upper - epochs_lower) * (
            (hidden_nodes_upper - hidden_nodes_lower) / hidden_nodes_step_rate) *
                           ((learning_rate_upper - learning_rate_lower) / learning_rate_step_rate))
    counter = manager.Value('i', 1)
    thread_safe_cache = manager.Namespace()
    thread_safe_cache.cache = cache

    parameter_list = helper.create_parameter_permutations(epochs_lower, epochs_upper, hidden_nodes_lower,
                                                          hidden_nodes_upper, hidden_nodes_step_rate,
                                                          learning_rate_lower, learning_rate_upper,
                                                          learning_rate_step_rate)
    pool = Pool()
    results = pool.starmap(threadable_training, [(*params, thread_safe_cache.cache, counter, total_iterations) for params in parameter_list])

    best_model, best_performance = determine_best_performing_model()
    print(f"Best performing model: {best_model} with a performance of {best_performance}")
    return extract_parameters(best_model)


def threadable_training(epochs, hidden_nodes, learning_rate, cache, counter, total_iterations: int):
    input_nodes: int = 784
    output_nodes: int = 10
    current_iteration = counter.value
    counter.value += 1
    if already_tested(cache, epochs, hidden_nodes, learning_rate):
        print(f"Already created a network with epochs={epochs}, "
              f"hidden_nodes={hidden_nodes}, learning_rate={learning_rate}")
        return
    ann, training_time = train_network(epochs, input_nodes, hidden_nodes, output_nodes, learning_rate)
    performance = test_network(ann)
    cache.add_entry(hidden_nodes, epochs, learning_rate, performance)
    name = generate_file_name(hidden_nodes, learning_rate, epochs, performance)
    # estimated_runtime = estimate_runtime_left(current_iteration, total_iterations, training_time)
    print(f"{name} - {current_iteration}/{total_iterations}")
    current_iteration += 1
    ann.export_to_file(name)


def main():
    with Manager() as manager:
        cache = SimpleCache()
        models = get_all_models()
        cache.add_all_from_filename_list(models)
        epochs_lower = 1
        epochs_upper = 2
        hidden_nodes_lower = 150
        hidden_nodes_upper = 170
        hidden_nodes_step_rate = 10
        learning_rate_lower = 12
        learning_rate_upper = 14
        learning_rate_step_rate = 1
        determine_best_parameters(epochs_lower, epochs_upper, hidden_nodes_lower, hidden_nodes_upper,
                                  hidden_nodes_step_rate, learning_rate_lower, learning_rate_upper,
                                  learning_rate_step_rate, cache, manager)


if __name__ == '__main__':
    main()
