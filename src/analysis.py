from main import *
import jsonpickle
import os


if __name__ == '__main__':

    parameter_names = sorted(os.listdir('../results/'))
    for parameter_name in parameter_names:
        parameter_values = sorted(os.listdir(f'../results/{parameter_name}'))

        for parameter_value in parameter_values:
            print(parameter_name, '=', parameter_value)

            scenes = []

            filename = f'../results/{parameter_name}/{parameter_value}'
            with open(filename, 'r') as f:
                results = jsonpickle.decode(f.read())

            for index, repetition in enumerate(results):
                print(f'rep {index + 1}: ', end='')
                for scene in repetition:
                    print(scene.graph().mst(), scene.graph().cost(), end='   ')
                print()

                scenes.append(repetition[0])

            visualise(scenes)

