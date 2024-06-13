from main import *
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


if __name__ == '__main__':

    # specify here which experiments you want to load in
    parameter_setups = {
        # 'initial_population_density': [0.01, 0.04],
        'reproduction_threshold': [30],
    }
    data = load_data('../results', parameter_setups)

    # scenes = data['initial_population_density']['0.04'][0]
    show_mst(data)

    visualise(data['reproduction_threshold']['30'][2])
