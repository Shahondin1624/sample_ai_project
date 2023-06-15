import numpy.random
from scipy import special

from helper import derive_file_path


class NeuralNetwork:
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int, learning_rate: float):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.weights_input_hidden = numpy.random.normal(0.0, pow(self.input_nodes, -0.5),
                                                        (self.hidden_nodes, self.input_nodes))
        self.weights_hidden_output = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5),
                                                         (self.output_nodes, self.hidden_nodes))
        self.activation_function = lambda x: special.expit(x)
        pass

    def train(self, input_list, target_list):
        # print("Transposing training data...")
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
        # print("Calculating hidden inputs...")
        hidden_inputs = numpy.dot(self.weights_input_hidden, inputs)
        # print("Calculating hidden outputs...")
        hidden_outputs = self.activation_function(hidden_inputs)
        # print("Calculating output inputs...")
        final_inputs = numpy.dot(self.weights_hidden_output, hidden_outputs)
        # print("Calculating output values...")
        final_outputs = self.activation_function(final_inputs)
        # print("Calculating total errors...")
        output_errors = targets - final_outputs
        # print("Calculating hidden errors...")
        hidden_errors = numpy.dot(self.weights_hidden_output.T, output_errors)
        # print("Adjusting weights (hidden->output)...")
        self.weights_hidden_output += self.learning_rate * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs))
        # print("Adjusting weights (input->hidden)...")
        self.weights_input_hidden += self.learning_rate * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs))

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.weights_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def print_internals(self):
        print(f"Weights input->hidden: {self.weights_input_hidden}")
        print(f"Weights hidden->output: {self.weights_hidden_output}")

    def export_to_file(self, name: str):
        file_path = derive_file_path(name)
        # savez_compressed accepts named parameters that are then stored in the file
        numpy.savez_compressed(file_path, weights_input_hidden=self.weights_input_hidden,
                               weights_hidden_output=self.weights_hidden_output, learning_rate=self.learning_rate)


def import_model(name: str):
    file_path = derive_file_path(name)
    data = numpy.load(file_path)
    weight_input_hidden = data['weights_input_hidden']
    weight_hidden_output = data['weights_hidden_output']
    learning_rate = data['learning_rate']
    ann = create_from_arrays(weight_input_hidden, weight_hidden_output, learning_rate)
    return ann


def create_from_arrays(weight_input_hidden: numpy.numarray, weight_hidden_output: numpy.numarray, learning_rate: float):
    dimensions_input_hidden = weight_input_hidden.shape
    dimensions_hidden_output = weight_hidden_output.shape
    input_nodes = dimensions_input_hidden[1]
    hidden_nodes = dimensions_input_hidden[0]
    output_nodes = dimensions_hidden_output[0]
    ann = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    ann.weights_input_hidden = weight_input_hidden
    ann.weights_hidden_output = weight_hidden_output
    return ann
