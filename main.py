import helper
from helper import extract_parameters, get_all_models, generate_file_name
from neuralNetwork import train_network, test_network
from simpleCache import SimpleCache


# checks whether a network with the given parameters has already been trained by checking the cache
def already_trained(cache: SimpleCache, epochs: int, hidden_nodes: int, learning_rate: float):
    return cache.contains_tuple(hidden_nodes, epochs, learning_rate)


# Will return the following values of the best performing model in this order: performance, epochs, hidden_nodes,
# learning_rate. lower/upper represent the bounds in between models will be tested. lower < upper for this code to work
# The upper bounds are inclusive
def determine_best_parameters(epochs_lower: int, epochs_upper: int, hidden_nodes_lower: int, hidden_nodes_upper: int,
                              hidden_nodes_step_rate: int, learning_rate_lower: int, learning_rate_upper: int,
                              learning_rate_step_rate: int, cache: SimpleCache):
    epochs_upper += 1
    hidden_nodes_upper += 1
    learning_rate_upper += 1
    total_iterations = int((epochs_upper - epochs_lower) * (
            (hidden_nodes_upper - hidden_nodes_lower) / hidden_nodes_step_rate) *
                           ((learning_rate_upper - learning_rate_lower) / learning_rate_step_rate))

    parameter_list = helper.create_parameter_permutations(epochs_lower, epochs_upper, hidden_nodes_lower,
                                                          hidden_nodes_upper, hidden_nodes_step_rate,
                                                          learning_rate_lower, learning_rate_upper,
                                                          learning_rate_step_rate)
    for counter, parameters in enumerate(parameter_list):
        epochs, hidden_nodes, learning_rate = parameters
        threadable_training(epochs, hidden_nodes, learning_rate, cache, counter + 1, total_iterations)

    best_model, best_performance = helper.determine_best_performing_model()
    print(f"Best performing model: {best_model} with a performance of {best_performance}")
    return extract_parameters(best_model)


# threadsafe implementation of a full training run of one network
def threadable_training(epochs, hidden_nodes, learning_rate, cache, current_iteration: int, total_iterations: int):
    input_nodes: int = 784
    output_nodes: int = 10
    if already_trained(cache, epochs, hidden_nodes, learning_rate):
        print(f"Already created a network with epochs={epochs}, "
              f"hidden_nodes={hidden_nodes}, learning_rate={learning_rate}")
        return
    ann, training_time = train_network(epochs, input_nodes, hidden_nodes, output_nodes, learning_rate)
    performance = test_network(ann)
    cache.add_entry(hidden_nodes, epochs, learning_rate, performance)
    name = generate_file_name(hidden_nodes, learning_rate, epochs, performance)
    # estimated_runtime = estimate_runtime_left(current_iteration, total_iterations, training_time)
    print(f"{name} - {current_iteration}/{total_iterations}")
    ann.export_to_file(name)


def main():
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
                              learning_rate_step_rate, cache)


if __name__ == '__main__':
    main()
