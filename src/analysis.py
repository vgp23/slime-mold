from main import *
import jsonpickle
import os
import time


def load_data(dirname, parameter_setups=None):
    data = dict()

    for parameter_name in sorted(os.listdir(dirname)):
        if parameter_name not in parameter_setups:
            continue

        data[parameter_name] = dict()

        for parameter_value in sorted(os.listdir(f'{dirname}/{parameter_name}')):
            if float(parameter_value) not in parameter_setups[parameter_name]:
                continue

            print('loading', parameter_name, '=', parameter_value)

            with open(f'{dirname}/{parameter_name}/{parameter_value}', 'r') as f:
                data[parameter_name][parameter_value] = jsonpickle.decode(f.read())

    return data


if __name__ == '__main__':

    # specify here which experiments you want to load in
    parameter_setups = {
        'initial_population_density': [0.01, 0.04, 0.07],
        'reproduction_threshold': [10, 15, 20],
    }
    data = load_data('../results', parameter_setups)

    for parameter_name in data:
        for parameter_value in data[parameter_name]:
            results = data[parameter_name][parameter_value]

            for index, repetition in enumerate(results):
                print(f'rep {index + 1}: ', end='')
                for scene in repetition:
                    print(scene.graph().mst(), scene.graph().cost(), end='   ')
                print()

