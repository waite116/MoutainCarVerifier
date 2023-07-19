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
    [-0.50000, -0.49975],
    [-0.49975, -0.49950],
    [-0.49950, -0.49925],
    [-0.49925, -0.49900],
    [-0.49900, -0.49875],
    [-0.49875, -0.49850],
    [-0.49850, -0.49825],
    [-0.49825, -0.49800],
    [-0.49800, -0.49775],
    [-0.49775, -0.49750],
    [-0.49750, -0.49725],
    [-0.49725, -0.49700],
    [-0.49700, -0.49675],
    [-0.49675, -0.49650],
    [-0.49650, -0.49625],
    [-0.49625, -0.49600],
    [-0.49600, -0.49575],
    [-0.49575, -0.49550],
    [-0.49550, -0.49525],
    [-0.49525, -0.49500],
    [-0.49500, -0.49475],
    [-0.49475, -0.49450],
    [-0.49450, -0.49425],
    [-0.49425, -0.49400],
    [-0.49400, -0.49375],
    [-0.49375, -0.49350],
    [-0.49350, -0.49325],
    [-0.49325, -0.49300],
    [-0.49300, -0.49275],
    [-0.49275, -0.49250],
    [-0.49250, -0.49225],
    [-0.49225, -0.49200],
    [-0.49200, -0.49175],
    [-0.49175, -0.49150],
    [-0.49150, -0.49125],
    [-0.49125, -0.49100],
    [-0.49100, -0.49075],
    [-0.49075, -0.49050],
    [-0.49050, -0.49025],
    [-0.49025, -0.49000],
    [-0.49000, -0.48975],
    [-0.48975, -0.48950],
    [-0.48950, -0.48925],
    [-0.48925, -0.48900],
    [-0.48900, -0.48975]
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
