import random


class Individual:

    @staticmethod
    def individual(icls):
        genome = list()
        genome.append(random.uniform(-1.5, 4))
        genome.append(random.uniform(-3, 4))

        return icls(genome)

    @staticmethod
    def svc(number_features, icls):
        genome = list()
        # kernel
        list_kernel = ["linear", "rbf", "poly", "sigmoid"]
        genome.append(list_kernel[random.randint(0, 3)])
        # c
        k = random.uniform(0.1, 80)
        genome.append(k)
        # degree
        genome.append(random.uniform(0.1, 5))
        # gamma
        gamma = random.uniform(0.001, 5)
        genome.append(gamma)
        # coeff
        coeff = random.uniform(0.01, 10)
        genome.append(coeff)

        # for i in range(0, number_features):
        #     genome.append(random.randint(0, 1))

        return icls(genome)

    @staticmethod
    def decision_tree_classifier(number_features, icls):
        genome = list()
        # criterion
        list_criterion = ["gini", "entropy"]
        genome.append(list_criterion[random.randint(0, 1)])
        # splitter
        list_splitter = ["best", "random"]
        genome.append(list_splitter[random.randint(0, 1)])
        # max_depth
        max_depth = random.randint(2, 10)
        genome.append(max_depth)
        # min_samples_split
        min_samples_split = random.randint(2, 10)
        genome.append(min_samples_split)
        # min_samples_leaf
        min_samples_leaf = random.randint(1, 10)
        genome.append(min_samples_leaf)
        # min_weight_fraction_leaf
        min_weight_fraction_leaf = random.uniform(0, 0.5)
        genome.append(min_weight_fraction_leaf)
        # max_features
        list_max_features = ["auto", "sqrt", "log2"]
        genome.append(list_max_features[random.randint(0, 2)])

        for i in range(0, number_features):
            genome.append(random.randint(0, 1))

        return icls(genome)

    @staticmethod
    def k_neighbors_classifier(number_features, icls):
        genome = list()
        # n_neighbors
        n_neighbors = random.randint(1, 10)
        genome.append(n_neighbors)
        # weights
        list_weights = ["uniform", "distance"]
        genome.append(list_weights[random.randint(0, 1)])
        # algorithm
        list_algorithm = ["auto", "ball_tree", "kd_tree", "brute"]
        genome.append(list_algorithm[random.randint(0, 3)])
        # leaf_size
        leaf_size = random.randint(1, 100)
        genome.append(leaf_size)
        # p
        p = random.randint(1, 5)
        genome.append(p)

        for i in range(0, number_features):
            genome.append(random.randint(0, 1))

        return icls(genome)

    @staticmethod
    def extra_tree_classifier(number_features, icls):
        genome = list()
        # criterion
        list_criterion = ["gini", "entropy"]
        genome.append(list_criterion[random.randint(0, 1)])
        # splitter
        list_splitter = ["random", "best"]
        genome.append(list_splitter[random.randint(0, 1)])
        # max_depth
        max_depth = random.randint(2, 10)
        genome.append(max_depth)
        # min_samples_split
        min_samples_split = random.randint(2, 10)
        genome.append(min_samples_split)
        # min_samples_leaf
        min_samples_leaf = random.randint(1, 10)
        genome.append(min_samples_leaf)
        # min_weight_fraction_leaf
        min_weight_fraction_leaf = random.uniform(0, 0.5)
        genome.append(min_weight_fraction_leaf)
        # max_features
        list_max_features = ["auto", "sqrt", "log2"]
        genome.append(list_max_features[random.randint(0, 2)])

        for i in range(0, number_features):
            genome.append(random.randint(0, 1))

        return icls(genome)

    @staticmethod
    def mlp_classifier(number_features, icls):
        genome = list()
        # hidden_layer_sizes
        hidden_layer_sizes = random.randint(50, 200)
        genome.append(hidden_layer_sizes)
        # activation
        list_activation = ["identity", "logistic", "tanh", "relu"]
        genome.append(list_activation[random.randint(0, 3)])
        # solver
        list_solver = ["lbfgs", "sgd", "adam"]
        genome.append(list_solver[random.randint(0, 2)])
        # alpha
        alpha = random.random()
        genome.append(alpha)
        #
        # for i in range(0, number_features):
        #     genome.append(random.randint(0, 1))

        return icls(genome)

    @staticmethod
    def random_forrest_classifier(number_features, icls):
        genome = list()
        # n_estimators
        n_estimators = random.randint(10, 100)
        genome.append(n_estimators)
        # criterion
        list_criterion = ["gini", "entropy"]
        genome.append(list_criterion[random.randint(0, 1)])
        # max_depth
        max_depth = random.randint(1, 100)
        genome.append(max_depth)
        # min_samples_split
        min_samples_split = random.randint(2, 50)
        genome.append(min_samples_split)
        # min_samples_leaf
        min_samples_leaf = random.randint(1, 50)
        genome.append(min_samples_leaf)
        # min_weight_fraction_leaf
        min_weight_fraction_leaf = random.uniform(0, .5)
        genome.append(min_weight_fraction_leaf)
        # max_features
        list_max_features = ["auto", "log2"]
        genome.append(list_max_features[random.randint(0, 1)])
        # max_leaf_nodes
        max_leaf_nodes = random.randint(2, 50)
        genome.append(max_leaf_nodes)

        for i in range(0, number_features):
            genome.append(random.randint(0, 1))

        return icls(genome)
