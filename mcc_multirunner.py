#!/usr/bin/python3

import os
import subprocess
import multiprocessing

verisig_path = '../verisig'
flowstar_path = '../flowstar/flowstar'
output_path = 'output_MCC'
xml_path = 'MCC_600.xml'
dnn1_yaml = 'networks/Generator2x50_100_Decay1Loss1.113e-06.yml'
dnn2_yaml = 'networks/Contraster1x50_Decay1Loss57.36.yml'
dnn3_yaml = 'networks/Image2StateRobuster1x20ImageSize20x30Decay0.001.yml'
dnn3_yaml = 'networks/ControllerSigmoid16x16.yml'

if not os.path.exists(output_path):
        os.mkdir(output_path)


legend = ['X1_LOWER', 'X1_UPPER']

test_set = [
    [-0.5000, -0.4995],
    [-0.4995, -0.4990],
    [-0.4990, -0.4985],
    [-0.4985, -0.4980],
    [-0.4980, -0.4975],
    [-0.4975, -0.4970],
    [-0.4970, -0.4965],
    [-0.4965, -0.4960],
    [-0.4960, -0.4955],
    [-0.4955, -0.4950],
    [-0.4950, -0.4945],
    [-0.4945, -0.4940],
    [-0.4940, -0.4935],
    [-0.4935, -0.4930],
    [-0.4930, -0.4925],
    [-0.4925, -0.4920],
    [-0.4920, -0.4915],
    [-0.4915, -0.4910],
    [-0.4910, -0.4905],
    [-0.4905, -0.4900],
    [-0.4900, -0.4895],
    [-0.4895, -0.4890],
    [-0.4890, -0.4885],
    [-0.4885, -0.4880],
    [-0.4880, -0.4875],
    [-0.4875, -0.4870],
    [-0.4870, -0.4865],
    [-0.4865, -0.4860],
    [-0.4860, -0.4855],
    [-0.4855, -0.4850],
    [-0.4850, -0.4845],
    [-0.4845, -0.4840],
    [-0.4840, -0.4835],
    [-0.4835, -0.4830],
    [-0.4830, -0.4825],
    [-0.4825, -0.4820],
    [-0.4820, -0.4815],
    [-0.4815, -0.4810],
    [-0.4810, -0.4805],
    [-0.4805, -0.4800]
]

print("Building the base model...")
subprocess.run([verisig_path, '-vc=MCC_multi.yml', '-o' ,'-nf', xml_path, dnn1_yaml, dnn2_yaml, dnn3_yaml, dnn4_yaml])

with open('MCC_600.model', 'r') as f:
    model = f.read()


#===========================================================================================
# Begin Parallel Function
#===========================================================================================
def evaluate_conditions(conditions):
    test_model = model
    for i in range(len(legend)):
        test_model = test_model.replace(legend[i], str(conditions[i]))

    with open(output_path + '/MCC_' + str(conditions[0]) + ', ' + str(conditions[1]) + '.txt', 'w') as f:
        subprocess.run(flowstar_path + ' ' + dnn1_yaml + ' ' + dnn2_yaml + ' ' + dnn3_yaml + ' ' + dnn4_yaml, input=test_model, shell=True, universal_newlines=True, stdout=f)
    print('Finished Interval: [' + str(conditions[0]) + ', ' + str(conditions[1]) + ']')
#===========================================================================================
# End Parallel Function
#===========================================================================================

print("Starting parallel verification")
num_parallel = 40
with multiprocessing.Pool(processes=num_parallel) as pool:
    pool.map(evaluate_conditions, test_set)
