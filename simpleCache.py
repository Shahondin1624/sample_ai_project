import helper


class SimpleCache:
    def __init__(self):
        self.cache = {}

    def add_entry(self, hidden_nodes: int, epochs: int, learning_rate: float, performance: float):
        self.cache[(hidden_nodes, epochs, learning_rate)] = performance

    def add_from_list(self, parameter_list: list):
        hidden_nodes = parameter_list[2]
        epochs = parameter_list[1]
        learning_rate = parameter_list[3]
        performance = parameter_list[0]
        self.add_entry(hidden_nodes, epochs, learning_rate, performance)

    def add_all_from_filename_list(self, filename_list: list):
        for entry in filename_list:
            parameters = helper.extract_parameters(entry)
            self.add_from_list(parameters)

    def contains_tuple(self, hidden_nodes: int, epochs: int, learning_rate: float):
        return self.cache.__contains__((hidden_nodes, epochs, learning_rate))

    # returns performance, model-parameters
    def get_best_performing_model(self):
        key_value_pair = max(self.cache.items(), key=lambda x: x[1])
        return key_value_pair[1], key_value_pair[0]
