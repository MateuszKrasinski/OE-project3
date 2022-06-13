import math

from matplotlib import pyplot as plt
from mealpy.swarm_based import CSA


def draw_plot_best(best_fitnesses):
    plt.figure()
    plt.xlabel("epochs")
    plt.ylabel("best fitness")
    plt.title("Best values on the next epoch")
    plt.plot([i for i in range(len(best_fitnesses))], best_fitnesses)
    plt.savefig('p100_a01/best_fitness.png')


def fitness_function(individual_value):
    result = math.sin((individual_value[0] + individual_value[1])) + math.pow(
        (individual_value[0] - individual_value[1]), 2) - 1.5 * individual_value[0] + 2.5 * individual_value[1] + 1

    return result


def draw_plot_avg(param):
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Average values")
    plt.title("Average values on the next epoch")
    plt.plot([i for i in range(len(param))], param)
    plt.savefig('p100_a01/average.png')


def draw_plot_sd(param):
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Standard deviation")
    plt.title("The standard deviation on the next epoch")
    plt.plot([i for i in range(len(param))], param)
    plt.savefig('p100_a01/standard_deviation.png')


problem_dict1 = {
    "fit_func": fitness_function,
    "lb": [-1.5, -3],
    "ub": [4, 4],
    "minmax": "min"
}

if __name__ == '__main__':
    model = CSA.BaseCSA(problem_dict1, epoch=1000, pop_size=100, p_a=0.1)
    best_position, best_fitness, best_fitnesses, avg, sd = model.solve()
    print(f"Best solution: {best_position}, Best fitness: {best_fitness}")
    draw_plot_best(best_fitnesses)
    draw_plot_sd(sd)
    draw_plot_avg(avg)
    model.history.save_global_best_fitness_chart(filename="p100_a01/gbfc")
    model.history.save_local_best_fitness_chart(filename="p100_a01/lbfc")

    model.history.save_diversity_chart(filename="p100_a01/dc")

    model.history.save_trajectory_chart(list_agent_idx=[1, 10, 50, 100], selected_dimensions=[2], filename="p100_a01/tc")
    model.history.save_global_objectives_chart(filename="p100_a01/goc")
    model.history.save_local_objectives_chart(filename="p100_a01/loc")
