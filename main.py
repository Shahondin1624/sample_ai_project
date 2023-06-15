import neuralNetwork
import numpy as numpy
import scipy.special as scipy
import matplotlib.pyplot as matplot
import dill
import time


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
    milliseconds = int((training_time - int(training_time)) * 1000)
    formatted_runtime = time.strftime(f"%H:%M:%S.{milliseconds:03d}", time.gmtime(training_time))
    print(f"Training took: {formatted_runtime}s")
    return ann


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
    return "performance_" + str(performance) + "_epochs_" + str(epochs) + "_hidden_nodes_" + str(hidden_nodes) + \
        "_learning_rate_" + str(learning_rate)


def main():
    input_nodes: int = 784
    output_nodes: int = 10

    for epochs in range(1, 11):
        for hidden_nodes in range(100, 260, 10):
            for learning_rate_as_int in range(10, 40, 5):
                learning_rate = learning_rate_as_int / 100.0
                ann = train_network(epochs, input_nodes, hidden_nodes, output_nodes, learning_rate)
                performance = test_network(ann)
                name = generate_file_name(hidden_nodes, learning_rate, epochs, performance)
                print(name)
                export_model(ann, name)


if __name__ == '__main__':
    main()
