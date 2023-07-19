#!/usr/bin/python3

import os
import subprocess
import multiprocessing

verisig_path = '../verisig'
flowstar_path = '../flowstar/flowstar'
output_path = 'output_MCC_0.48900_0.47500'
xml_path = 'MCC_600.xml'
dnn1_yaml = 'networks/Generator2x50_100_Decay1Loss1.113e-06.yml'
dnn2_yaml = 'networks/Contraster1x50_Decay1Loss57.36.yml'
dnn3_yaml = 'networks/Image2StateRobuster1x20ImageSize20x30Decay0.001.yml'
dnn4_yaml = 'networks/ControllerSigmoid16x16.yml'

if not os.path.exists(output_path):
        os.mkdir(output_path)


legend = ['X1_LOWER', 'X1_UPPER']

test_set = [
    [-0.48900, -0.48875],
    [-0.48875, -0.48850],
    [-0.48850, -0.48825],
    [-0.48825, -0.48800],
    [-0.48800, -0.48775],
    [-0.48775, -0.48750],
    [-0.48750, -0.48725],
    [-0.48725, -0.48600],
    [-0.48600, -0.48575],
    [-0.48575, -0.48550],
    [-0.48550, -0.48525],
    [-0.48525, -0.48500],
    [-0.48500, -0.48475],
    [-0.48475, -0.48450],
    [-0.48450, -0.48425],
    [-0.48425, -0.48400],
    [-0.48400, -0.48375],
    [-0.48375, -0.48350],
    [-0.48350, -0.48325],
    [-0.48325, -0.48300],
    [-0.48300, -0.48275],
    [-0.48275, -0.48250],
    [-0.48250, -0.48225],
    [-0.48225, -0.48200],
    [-0.48200, -0.48175],
    [-0.48175, -0.48150],
    [-0.48150, -0.48125],
    [-0.48125, -0.48100],
    [-0.48100, -0.48075],
    [-0.48075, -0.48050],
    [-0.48050, -0.48025],
    [-0.48025, -0.48000],
    [-0.48000, -0.47975],
    [-0.47975, -0.47950],
    [-0.47950, -0.47925],
    [-0.47925, -0.47900],
    [-0.47900, -0.47875],
    [-0.47875, -0.47850],
    [-0.47850, -0.47825],
    [-0.47825, -0.47800],
    [-0.47800, -0.47775],
    [-0.47775, -0.47750],
    [-0.47750, -0.47725],
    [-0.47725, -0.47700],
    [-0.47700, -0.47675],
    [-0.47675, -0.47650],
    [-0.47650, -0.47625],
    [-0.47625, -0.47600],
    [-0.47600, -0.47575],
    [-0.47575, -0.47550],
    [-0.47550, -0.47525],
    [-0.47525, -0.47500]
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
