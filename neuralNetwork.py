import numpy.random
from scipy import special


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
        # print(f"Training for a dataset of {len(input_list)} items")
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

    def export_internals(self):
        return self.weights_input_hidden, self.weights_hidden_output
