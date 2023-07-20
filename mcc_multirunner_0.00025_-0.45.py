#!/usr/bin/python3

import os
import subprocess
import multiprocessing

verisig_path = '../verisig'
flowstar_path = '../flowstar/flowstar'
output_path = 'output_MCC_0.4500_0.43600'
xml_path = 'MCC_600.xml'
dnn1_yaml = 'networks/Generator2x50_100_Decay1Loss1.113e-06.yml'
dnn2_yaml = 'networks/Contraster1x50_Decay1Loss57.36.yml'
dnn3_yaml = 'networks/Image2StateRobuster1x20ImageSize20x30Decay0.001.yml'
dnn4_yaml = 'networks/ControllerSigmoid16x16.yml'

if not os.path.exists(output_path):
        os.mkdir(output_path)


legend = ['X1_LOWER', 'X1_UPPER']

test_set = [
    [-0.45000, -0.44975],
    [-0.44975, -0.44950],
    [-0.44950, -0.44925],
    [-0.44925, -0.44900],
    [-0.44900, -0.44875],
    [-0.44875, -0.44850],
    [-0.44850, -0.44825],
    [-0.44825, -0.44800],
    [-0.44800, -0.44775],
    [-0.44775, -0.44750],
    [-0.44750, -0.44725],
    [-0.44725, -0.44700],
    [-0.44700, -0.44675],
    [-0.44675, -0.44650],
    [-0.44650, -0.44625],
    [-0.44625, -0.44600],
    [-0.44600, -0.44575],
    [-0.44575, -0.44550],
    [-0.44550, -0.44525],
    [-0.44525, -0.44500],
    [-0.44500, -0.44475],
    [-0.44475, -0.44450],
    [-0.44450, -0.44425],
    [-0.44425, -0.44400],
    [-0.44400, -0.44375],
    [-0.44375, -0.44350],
    [-0.44350, -0.44325],
    [-0.44325, -0.44300],
    [-0.44300, -0.44275],
    [-0.44275, -0.44250],
    [-0.44250, -0.44225],
    [-0.44225, -0.44200],
    [-0.44200, -0.44175],
    [-0.44175, -0.44150],
    [-0.44150, -0.44125],
    [-0.44125, -0.44100],
    [-0.44100, -0.44075],
    [-0.44075, -0.44050],
    [-0.44050, -0.44025],
    [-0.44025, -0.44000],
    [-0.44000, -0.43975],
    [-0.43975, -0.43950],
    [-0.43950, -0.43925],
    [-0.43925, -0.43900],
    [-0.43900, -0.43875],
    [-0.43875, -0.43850],
    [-0.43850, -0.43825],
    [-0.43825, -0.43800],
    [-0.43800, -0.43775],
    [-0.43775, -0.43750],
    [-0.43750, -0.43725],
    [-0.43725, -0.43700],
    [-0.43700, -0.43675],
    [-0.43675, -0.43650],
    [-0.43650, -0.43625],
    [-0.43625, -0.43600]
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
