#!/usr/bin/python3

import os
import subprocess
import multiprocessing

verisig_path = '../verisig'
flowstar_path = '../flowstar/flowstar'
output_path = 'output_MCC_0.40800_0.40000'
xml_path = 'MCC_600.xml'
dnn1_yaml = 'networks/Generator2x50_100_Decay1Loss1.113e-06.yml'
dnn2_yaml = 'networks/Contraster1x50_Decay1Loss57.36.yml'
dnn3_yaml = 'networks/Image2StateRobuster1x20ImageSize20x30Decay0.001.yml'
dnn4_yaml = 'networks/ControllerSigmoid16x16.yml'

if not os.path.exists(output_path):
        os.mkdir(output_path)


legend = ['X1_LOWER', 'X1_UPPER']

test_set = [
    [-0.40800, -0.40775],
    [-0.40775, -0.40750],
    [-0.40750, -0.40725],
    [-0.40725, -0.40700],
    [-0.40700, -0.40675],
    [-0.40675, -0.40650],
    [-0.40650, -0.40625],
    [-0.40625, -0.40600],
    [-0.40600, -0.40575],
    [-0.40575, -0.40550],
    [-0.40550, -0.40525],
    [-0.40525, -0.40500],
    [-0.40500, -0.40475],
    [-0.40475, -0.40450],
    [-0.40450, -0.40425],
    [-0.40425, -0.40400],
    [-0.40400, -0.40375],
    [-0.40375, -0.40350],
    [-0.40350, -0.40325],
    [-0.40325, -0.40300],
    [-0.40300, -0.40275],
    [-0.40275, -0.40250],
    [-0.40250, -0.40225],
    [-0.40225, -0.40200],
    [-0.40200, -0.40175],
    [-0.40175, -0.40150],
    [-0.40150, -0.40125],
    [-0.40125, -0.40100],
    [-0.40100, -0.40075],
    [-0.40075, -0.40050],
    [-0.40050, -0.40025],
    [-0.40025, -0.40000]
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
num_parallel = len(test_set)
with multiprocessing.Pool(processes=num_parallel) as pool:
    pool.map(evaluate_conditions, test_set)
