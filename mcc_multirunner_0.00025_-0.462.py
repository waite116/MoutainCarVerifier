#!/usr/bin/python3

import os
import subprocess
import multiprocessing

verisig_path = '../verisig'
flowstar_path = '../flowstar/flowstar'
output_path = 'output_MCC_0.46200_0.45000'
xml_path = 'MCC_600.xml'
dnn1_yaml = 'networks/Generator2x50_100_Decay1Loss1.113e-06.yml'
dnn2_yaml = 'networks/Contraster1x50_Decay1Loss57.36.yml'
dnn3_yaml = 'networks/Image2StateRobuster1x20ImageSize20x30Decay0.001.yml'
dnn4_yaml = 'networks/ControllerSigmoid16x16.yml'

if not os.path.exists(output_path):
        os.mkdir(output_path)


legend = ['X1_LOWER', 'X1_UPPER']

test_set = [
    [-0.46200, -0.46175],
    [-0.46175, -0.46150],
    [-0.46150, -0.46125],
    [-0.46125, -0.46100],
    [-0.46100, -0.46075],
    [-0.46075, -0.46050],
    [-0.46050, -0.46025],
    [-0.46025, -0.46000],
    [-0.46000, -0.45975],
    [-0.45975, -0.45950],
    [-0.45950, -0.45925],
    [-0.45925, -0.45900],
    [-0.45900, -0.45875],
    [-0.45875, -0.45850],
    [-0.45850, -0.45825],
    [-0.45825, -0.45800],
    [-0.45800, -0.45775],
    [-0.45775, -0.45750],
    [-0.45750, -0.45725],
    [-0.45725, -0.45700],
    [-0.45700, -0.45675],
    [-0.45675, -0.45650],
    [-0.45650, -0.45625],
    [-0.45625, -0.45600],
    [-0.45600, -0.45575],
    [-0.45575, -0.45550],
    [-0.45550, -0.45525],
    [-0.45525, -0.45500],
    [-0.45500, -0.45475],
    [-0.45475, -0.45450],
    [-0.45450, -0.45425],
    [-0.45425, -0.45400],
    [-0.45400, -0.45375],
    [-0.45375, -0.45350],
    [-0.45350, -0.45325],
    [-0.45325, -0.45300],
    [-0.45300, -0.45275],
    [-0.45275, -0.45250],
    [-0.45250, -0.45225],
    [-0.45225, -0.45200],
    [-0.45200, -0.45175],
    [-0.45175, -0.45150],
    [-0.45150, -0.45125],
    [-0.45125, -0.45100],
    [-0.45100, -0.45075],
    [-0.45075, -0.45050],
    [-0.45050, -0.45025],
    [-0.45025, -0.45000]
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
