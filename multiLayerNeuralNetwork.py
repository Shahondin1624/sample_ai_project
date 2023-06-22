import numpy
from scipy import special

from helper import derive_file_path


class MultiLayerNeuralNetwork:
    def __init__(self, input_nodes: int, hidden_layer_nodes: list, output_nodes: int, learning_rate: float):
        self.input_nodes = input_nodes
        self.hidden_layer_nodes = hidden_layer_nodes
        self.hidden_layers = []
        self.__init_hidden_layers()
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.activation_function = lambda x: special.expit(x)

    def __init_hidden_layers(self):
        current_node_count = self.input_nodes
        for node_count in self.hidden_layer_nodes:
            weights_current_next = numpy.random.normal(0.0, pow(current_node_count, -0.5),
                                                       (node_count, current_node_count))
            self.hidden_layers.append(weights_current_next)
            current_node_count = node_count
        weights_current_next = numpy.random.normal(0.0, pow(current_node_count, -0.5),
                                                   (self.output_nodes, current_node_count))
        self.hidden_layers.append(weights_current_next)

    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # Forward pass
        layer_inputs = [inputs]
        layer_outputs = []
        for weights in self.hidden_layers:
            hidden_inputs = numpy.dot(weights, layer_inputs[-1])
            hidden_outputs = self.activation_function(hidden_inputs)
            layer_inputs.append(hidden_outputs)
            layer_outputs.append(hidden_outputs)

        final_outputs = layer_outputs[-1]
        output_errors = targets - final_outputs

        # Backward pass
        errors = output_errors
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            weights = self.hidden_layers[i]
            outputs = layer_outputs[i]
            inputs = layer_inputs[i]
            weights += self.learning_rate * numpy.dot((errors * outputs * (1.0 - outputs)), numpy.transpose(inputs))
            errors = numpy.dot(weights.T, errors)

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T

        # Forward pass
        layer_inputs = inputs
        for weights in self.hidden_layers:
            hidden_inputs = numpy.dot(weights, layer_inputs)
            hidden_outputs = self.activation_function(hidden_inputs)
            layer_inputs = hidden_outputs

        final_outputs = layer_inputs
        return final_outputs

    def print_internals(self):
        for weights in self.hidden_layers:
            print(f"Weights: {weights}")

    def export_to_file(self, name: str):
        file_path = derive_file_path(name)
        data = {'learning_rate': self.learning_rate}
        for i, weights in enumerate(self.hidden_layers):
            data[f'weights_{i}'] = weights
        numpy.savez_compressed(file_path, **data)


@staticmethod
def import_from_file(name: str):
    file_path = derive_file_path(name)
    data = numpy.load(file_path)
    learning_rate = data['learning_rate']
    weights = []
    i = 0
    while f'weights_{i}' in data:
        weights.append(data[f'weights_{i}'])
        i += 1
    input_nodes = weights[0].shape[1]
    hidden_layer_nodes = [w.shape[0] for w in weights[:-1]]
    output_nodes = weights[-1].shape[0]
    ann = MultiLayerNeuralNetwork(input_nodes, hidden_layer_nodes, output_nodes, learning_rate)
    ann.hidden_layers = weights
    return ann
