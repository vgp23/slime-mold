from main import *
import matplotlib.pyplot as plt
import jsonpickle
import os
import time
import pathlib


def load_data(dirname, parameter_setups=None):
    data = dict()

    for parameter_name in sorted(os.listdir(dirname)):
        if parameter_setups is not None and parameter_name not in parameter_setups:
            continue

        data[parameter_name] = dict()

        for parameter_value in sorted(map(float, os.listdir(f'{dirname}/{parameter_name}'))):
            if (
                parameter_setups is not None and
                parameter_value not in parameter_setups[parameter_name]
            ):
                continue

            print('loading', parameter_name, '=', parameter_value)

            with open(f'{dirname}/{parameter_name}/{parameter_value:g}', 'r') as f:
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


def compute_means_reps(graphs, measures):
    """Compute the means per repetition for each of the measures."""
    measures_means_reps = {measure: [] for measure in measures}  # save the repetition means

    for food_setup in range(len(graphs[0])):
        measure_datas = {measure: [] for measure in measures}

        for repetition in range(len(graphs)):
            graph = graphs[repetition][food_setup]

            if graph.connected:
                for measure, measure_info in measures.items():
                    measure_datas[measure].append(measure_info['f'](graph))

        for measure, measure_data in measure_datas.items():
            if len(measure_data) > 0:
                measures_means_reps[measure].append(sum(measure_data) / len(measure_data))

    return measures_means_reps


def compute_means(graphs, measures):
    """Compute the means and stds per food_setup for each of the measures."""
    measures_means_reps = compute_means_reps(graphs, measures)
    return {
        measure: (np.mean(measure_means), np.std(measure_means))
        for measure, measure_means in measures_means_reps.items()
    }


def compute_connectedness(graphs):
    return sum(sum(
        [[int(graph.connected) for graph in repetition] for repetition in graphs], []
    )) / (len(graphs) * len(graphs[0]))


def scenes_to_graphs(results):
    return [[scene.graph() for scene in repetition] for repetition in results]


def check_enough(graphs):
    count = 0
    for repetition in graphs:
        for graph in repetition:
            if graph.connected:
                count += 1
    return count < len(graphs) * len(graphs[0]) * 0.7


def plot_oneline(data, parameter_name, means, stds, label, color):
    plt.fill_between(means.keys(),
        np.array(list(means.values())) + np.array(list(stds.values())),
        np.array(list(means.values())) - np.array(list(stds.values())),
        alpha=0.35, lw=0, color=color
    )

    plt.scatter(means.keys(), means.values(), color=color)
    plt.plot(means.keys(), means.values(), color=color, label=label)

    plt.xlabel(parameter_name)
    plt.xlim([min(data[parameter_name]), max(data[parameter_name])])
    plt.ylim([0, 1])


def plot_connected(data, parameter_name, total_connected, color):
    values = list(data[parameter_name].keys())

    plt.scatter(values, total_connected, color=color,)
    plt.plot(values, total_connected, color=color, label='% connected')

    plt.xlabel(parameter_name)
    plt.xlim([min(data[parameter_name]), max(data[parameter_name])])
    plt.ylim([0, 1])


def plot_all(data, measures):
    for parameter_name in data:
        final_measures_means_stds = {measure: (dict(), dict(), info) for measure, info in measures.items()}
        total_connected = []

        for parameter_value in data[parameter_name]:
            print('processing', parameter_name, '=', parameter_value)
            results = data[parameter_name][parameter_value]
            graphs = scenes_to_graphs(results)

            total_connected.append(compute_connectedness(graphs))

            if check_enough(graphs):
                print('not enough data for this batch')
                continue

            measures_mean_std = compute_means(graphs, measures)
            for measure, (mean, std) in measures_mean_std.items():
                final_measures_means_stds[measure][0][parameter_value] = mean
                final_measures_means_stds[measure][1][parameter_value] = std

        # plot all data
        for measure, (means, stds, info) in final_measures_means_stds.items():
            plot_oneline(data, parameter_name, means, stds, info['name'], info['color'])
        plot_connected(data, parameter_name, total_connected, 'tab:purple')

        if not 'starvation_penalty' in data or parameter_name == 'starvation_penalty':
            plt.legend()

        # save the figure
        pathlib.Path('../figures').mkdir(parents=True, exist_ok=True)
        plt.savefig('../figures/' + parameter_name + '.png', dpi=300, bbox_inches='tight')

        plt.close()
        # plt.show()


def plot_hist(data, measures):
    for parameter_name in data:
        final_measures_means_reps = {measure: (dict(), info) for measure, info in measures.items()}

        for parameter_value in data[parameter_name]:
            print('processing', parameter_name, '=', parameter_value)
            results = data[parameter_name][parameter_value]
            graphs = scenes_to_graphs(results)

            if check_enough(graphs):
                print('not enough data for this batch')
                continue

            measures_means_reps = compute_means_reps(graphs, measures)
            for measure, means_reps in measures_means_reps.items():
                final_measures_means_reps[measure][0][parameter_value] = means_reps

        for measure, (all_value_data, info) in final_measures_means_reps.items():
            for parameter_value, value_data in all_value_data.items():
                plt.hist(value_data, bins=5)
                plt.xlabel(f'{info['name']}, {parameter_name} = {parameter_value}')
                plt.show()


if __name__ == '__main__':
    measures = {
        'cost': {
            'f': lambda graph: graph.mst_perfect() / graph.cost(),
            'name': '1 / $\\mathrm{TL_{MRST}}$',
            'color': 'tab:blue',
        },
        'performance': {
            'f': lambda graph: graph.mst_perfect() / graph.mst_actual(),
            'name': '$\\mathrm{MD_{MRST}}$',
            'color': 'tab:orange'
        },
        'fault tolerance': {
            'f': lambda graph: graph.fault_tolerance(),
            'name': '$\\mathrm{FT}$',
            'color': 'tab:green'
        },
        'efficiency': {
            'f': lambda graph: graph.fault_tolerance() / (graph.cost() / graph.mst_perfect()),
            'name': '$\\mathrm{FT}\\ /\\ \\mathrm{TL_{MRST}}$',
            'color': 'tab:red'
        },
    }

    # specify here which experiments you want to load in
    # parameter_setups = {
    #     'initial_population_density': [.01, .04, .07, .1, .3, .5],
    #     # 'elimination_threshold': [-5, -10, -15, -20, -25, -30]
    # }
    data = load_data('../results')#, parameter_setups)
    # plot_all(data, measures)
    plot_hist(data, measures)
