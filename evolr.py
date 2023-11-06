import argparse
import numpy as np


def _load_X(path):
    # Load the data.
    mat = np.loadtxt(path, dtype=int)
    max_doc_id = mat[:, 0].max()
    max_word_id = mat[:, 1].max()
    X = np.zeros(shape=(max_doc_id, max_word_id))
    for (docid, wordid, count) in mat:
        X[docid - 1, wordid - 1] = count
    return X


def _load_train(data, labels):
    # Load the labels.
    y = np.loadtxt(labels, dtype=int)
    X = _load_X(data)

    # Return.
    return [X, y]


def generate_initial_population(population_size, feature_count):
    return np.random.randn(population_size, feature_count)


def fitness_function(X, y, weights):
    predictions = np.dot(X, weights) > 0  # Using 0 as the threshold for classification
    accuracy = np.mean(predictions == y)
    return accuracy


def tournament_selection(population, fitnesses, survival_rate):
    selected_count = int(len(population) * survival_rate)
    selected_indices = []
    for _ in range(selected_count):
        # Randomly select tournament participants
        tournament_indices = np.random.choice(len(population), size=3, replace=False)
        tournament_fitnesses = fitnesses[tournament_indices]
        # Select the best individual from the tournament
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        selected_indices.append(winner_index)
    return population[selected_indices]


def crossover(parent1, parent2):
    cut_point = np.random.randint(1, len(parent1))
    offspring1 = np.concatenate((parent1[:cut_point], parent2[cut_point:]))
    offspring2 = np.concatenate((parent2[:cut_point], parent1[cut_point:]))
    return offspring1, offspring2


def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.randn()
    return individual


def create_next_generation(selected_population, population_size, mutation_rate):
    next_generation = []
    while len(next_generation) < population_size:
        # Randomly select two parents for crossover
        parent_indices = np.random.choice(len(selected_population), size=2, replace=False)
        parent1, parent2 = selected_population[parent_indices]
        offspring1, offspring2 = crossover(parent1, parent2)
        # Mutate the offspring
        offspring1 = mutate(offspring1, mutation_rate)
        offspring2 = mutate(offspring2, mutation_rate)
        # Add offspring to the next generation
        next_generation.append(offspring1)
        if len(next_generation) < population_size:
            next_generation.append(offspring2)

    return np.array(next_generation)


def predict_outcomes(X, weights, threshold=0.5):
    # Compute the dot product of features and weights
    raw_predictions = np.dot(X, weights)

    # Apply threshold to get binary outcomes
    predictions = (raw_predictions > threshold).astype(int)

    return predictions


def evolution_with_prediction(X, y, population_size, survival_rate, mutation_rate, generations):
    # Generate initial population
    population = generate_initial_population(population_size, X.shape[1])
    best_fitness = 0
    best_weights = None

    # Evolutionary loop
    for generation in range(generations):
        # Compute fitness for each individual
        fitnesses = np.array([fitness_function(X, y, individual) for individual in population])

        # Check for the best individual in this generation
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_weights = population[max_fitness_idx]

        # Select individuals based on fitness for reproduction
        selected_population = tournament_selection(population, fitnesses, survival_rate)

        # Create the next generation
        population = create_next_generation(selected_population, population_size, mutation_rate)

        # Report progress
        # print(f"Generation {generation}: Best fitness = {best_fitness}")

    # Use the best weights to make predictions
    predictions = predict_outcomes(X, best_weights)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homework 4",
                                     epilog="CSCI 4360/6360 Data Science II",
                                     add_help="How to use",
                                     prog="python homework4.py [train-data] [train-label] [test-data] <optional args>")
    parser.add_argument("paths", nargs=3)
    parser.add_argument("-n", "--population", default=150, type=int,
                        help="Population size [DEFAULT: 100].")
    parser.add_argument("-s", "--survival", default=0.3, type=float,
                        help="Per-generation survival rate [DEFAULT: 0.3].")
    parser.add_argument("-m", "--mutation", default=0.01, type=float,
                        help="Point mutation rate [DEFAULT: 0.01].")
    parser.add_argument("-g", "--generations", default=200, type=int,
                        help="Number of generations to run [DEFAULT: 100].")
    parser.add_argument("-r", "--random", default=-1, type=int,
                        help="Random seed for debugging [DEFAULT: -1].")
    args = vars(parser.parse_args())

    # Do we set a random seed?
    if args['random'] > -1:
        np.random.seed(args['random'])

    # Read in the training data.
    X, y = _load_train(args["paths"][0], args["paths"][1])

    # Run the evolutionary algorithm
    predictions = evolution_with_prediction(
        X, y,
        population_size=args['population'],
        survival_rate=args['survival'],
        mutation_rate=args['mutation'],
        generations=args['generations']
    )

    # Output the best weights found
    # print("Best weights found:", best_weights)
    # Output predictions for each unique document ID
    for doc_id, prediction in enumerate(predictions, start=1):
        print(prediction)
