from main import *
import matplotlib.pyplot as plt
import jsonpickle
import os
import time


def load_data(dirname, parameter_setups=None):
    data = dict()

    for parameter_name in sorted(os.listdir(dirname)):
        if parameter_setups is not None and parameter_name not in parameter_setups:
            continue

        data[parameter_name] = dict()

        for parameter_value in sorted(os.listdir(f'{dirname}/{parameter_name}')):
            if (
                parameter_setups is not None and
                float(parameter_value) not in parameter_setups[parameter_name]
            ):
                continue

            print('loading', parameter_name, '=', parameter_value)

            with open(f'{dirname}/{parameter_name}/{parameter_value}', 'r') as f:
                data[parameter_name][parameter_value] = jsonpickle.decode(f.read())

    return data


def show_connectedness(data):
    for parameter_name in data:
        for parameter_value in data[parameter_name]:
            results = data[parameter_name][parameter_value]

            print(parameter_name, '=', parameter_value)

            for index, repetition in enumerate(results):
                print(f'rep {index + 1}: ', end='')
                for scene in repetition:
                    print(
                        str(scene.graph().connected)[0],
                        str(scene.graph().fullyconnected(only_foods=True))[0],
                        end='   '
                    )
                print()


def show_mst(data):
    for parameter_name in data:
        for parameter_value in data[parameter_name]:
            results = data[parameter_name][parameter_value]

            print(parameter_name, '=', parameter_value)

            for index, repetition in enumerate(results):
                print(f'rep {index + 1}: ', end='')
                for scene in repetition:
                    graph = scene.graph()

                    print(graph.mst_perfect(), end=' ')
                    if graph.connected:
                        print(graph.mst_actual(), end=' ')
                    else:
                        print('   ', end='')
                    print('  ', end='')

                print()


def check_enough(results):
    count = 0
    for repetition in results:
        for scene in repetition:
            if scene.graph().connected:
                count += 1
    return count < len(results) * len(results[0]) * 0.7


def compute_means(results):
    cost_means, perf_means, ftol_means = [], [], []

    for s in range(len(results[0])):
        for r in range(len(results)):
            scene = results[r][s]
            graph = scene.graph()

            cost_valid_reps, perf_valid_reps, ftol_valid_reps = [], [], []

            if graph.connected:
                cost_valid_reps.append(graph.mst_perfect() / graph.cost())
                perf_valid_reps.append(graph.mst_perfect() / graph.mst_actual())
                ftol_valid_reps.append(graph.fault_tolerance())

            if len(cost_valid_reps) > 0:
                cost_means.append(sum(cost_valid_reps) / len(cost_valid_reps))
                perf_means.append(sum(perf_valid_reps) / len(perf_valid_reps))
                ftol_means.append(sum(ftol_valid_reps) / len(ftol_valid_reps))

    return np.array(cost_means), np.array(perf_means), np.array(ftol_means)


def plot_oneline(means, stds, label, parameter_name, color):
    plt.fill_between(means.keys(),
        np.array(list(means.values())) + np.array(list(stds.values())),
        np.array(list(means.values())) - np.array(list(stds.values())),
        alpha=0.35, lw=0, color=color
    )

    plt.scatter(means.keys(), means.values(), color=color)
    plt.plot(means.keys(), means.values(), color=color, label=label)

    plt.xlabel(parameter_name)
    plt.xlim([
        min(map(float, data[parameter_name])),
        max(map(float, data[parameter_name]))
    ])


def add_means(means_dict, stds_dict, means, parameter_value):
    means_dict[float(parameter_value)] = np.mean(means)
    stds_dict[float(parameter_value)] = np.std(means)


def plot_all(data):
    for parameter_name in data:
        cost_means, cost_stds = {}, {}
        perf_means, perf_stds = {}, {}
        ftol_means, ftol_stds = {}, {}

        for parameter_value in data[parameter_name]:
            print('processing', parameter_name, '=', parameter_value)
            results = data[parameter_name][parameter_value]

            if check_enough(results):
                print('not enough data for this batch')
                continue

            means_cost, means_perf, means_ftol = compute_means(results)

            add_means(cost_means, cost_stds, means_cost, parameter_value)
            add_means(perf_means, perf_stds, means_perf, parameter_value)
            add_means(ftol_means, ftol_stds, means_ftol, parameter_value)

        plot_oneline(cost_means, cost_stds, '$1 / \\mathrm{TL_{MRST}}$', parameter_name, 'tab:blue')
        plot_oneline(perf_means, perf_stds, '$\\mathrm{MD_{MRST}}$', parameter_name, 'tab:orange')
        plot_oneline(ftol_means, ftol_stds, '$\\mathrm{FT}$', parameter_name, 'tab:green')

        plt.legend()
        plt.show()


if __name__ == '__main__':
    # specify here which experiments you want to load in
    # parameter_setups = {
    #     'initial_population_density': [0.01, 0.04, 0.07, .1, .3, .5],
    #     # 'reproduction_threshold': [30],
    # }
    data = load_data('../results')#, parameter_setups)
    plot_all(data)