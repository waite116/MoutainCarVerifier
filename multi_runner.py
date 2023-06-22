#!/usr/bin/python3

import os
import subprocess
import multiprocessing

verisig_path = '../verisig'
flowstar_path = '../flowstar/flowstar'
output_path = 'output'
dnn1_yaml = 'networks/State2Image1x10ImageSize5x15.yml'
dnn2_yaml = 'networks/Image2State1x10ImageSize5x15.yml'
dnn3_yaml = 'networks/ControllerSigmoid16x16.yml'

if not os.path.exists(output_path):
        os.mkdir(output_path)


legend = ['X1_LOWER', 'X1_UPPER']

test_set = [
    [-0.60, -0.59],
    [-0.59, -0.585],
    [-0.585, -0.58],
    [-0.46, -0.45],
    [-0.45, -0.44],
    [-0.44, -0.43],
    [-0.43, -0.42],
    [-0.42, -0.41],
    [-0.41, -0.40]
]

print("Building the base model...")
subprocess.run([verisig_path, '-vc=MC_multi.yml', '-o' ,'-nf', 'MC75.xml', dnn1_yaml, dnn2_yaml, dnn3_yaml])

with open('MC75.model', 'r') as f:
    model = f.read()


#===========================================================================================
# Begin Parallel Function
#===========================================================================================
def evaluate_conditions(conditions):
    test_model = model
    for i in range(len(legend)):
        test_model = test_model.replace(legend[i], str(conditions[i]))

    with open(output_path + '/MC_' + str(conditions[0]) + '.txt', 'w') as f:
        subprocess.run(flowstar_path + ' ' + dnn1_yaml + ' ' + dnn2_yaml + ' ' + dnn3_yaml , input=test_model, shell=True, universal_newlines=True, stdout=f)
#===========================================================================================
# End Parallel Function
#===========================================================================================

print("Starting parallel verification")
num_parallel = multiprocessing.cpu_count() // 2
with multiprocessing.Pool(processes=num_parallel) as pool:
    pool.map(evaluate_conditions, test_set)
