#!/usr/bin/python3

import os
import subprocess
import multiprocessing

verisig_path = '../verisig'
flowstar_path = '../flowstar/flowstar'
output_path = 'output_MCC_0.54900_0.53500'
xml_path = 'MCC_600.xml'
dnn1_yaml = 'networks/Generator2x50_100_Decay1Loss1.113e-06.yml'
dnn2_yaml = 'networks/Contraster1x50_Decay1Loss57.36.yml'
dnn3_yaml = 'networks/Image2StateRobuster1x20ImageSize20x30Decay0.001.yml'
dnn4_yaml = 'networks/ControllerSigmoid16x16.yml'

if not os.path.exists(output_path):
        os.mkdir(output_path)


legend = ['X1_LOWER', 'X1_UPPER']

test_set = [
    [-0.54900, -0.54875],
    [-0.54875, -0.54850],
    [-0.54850, -0.54825],
    [-0.54825, -0.54800],
    [-0.54800, -0.54775],
    [-0.54775, -0.54750],
    [-0.54750, -0.54725],
    [-0.54725, -0.54700],
    [-0.54700, -0.54675],
    [-0.54675, -0.54650],
    [-0.54650, -0.54625],
    [-0.54625, -0.54600],
    [-0.54600, -0.54575],
    [-0.54575, -0.54550],
    [-0.54550, -0.54525],
    [-0.54525, -0.54500],
    [-0.54500, -0.54475],
    [-0.54475, -0.54450],
    [-0.54450, -0.54425],
    [-0.54425, -0.54400],
    [-0.54400, -0.54375],
    [-0.54375, -0.54350],
    [-0.54350, -0.54325],
    [-0.54325, -0.54300],
    [-0.54300, -0.54275],
    [-0.54275, -0.54250],
    [-0.54250, -0.54225],
    [-0.54225, -0.54200],
    [-0.54200, -0.54175],
    [-0.54175, -0.54150],
    [-0.54150, -0.54125],
    [-0.54125, -0.54100],
    [-0.54100, -0.54075],
    [-0.54075, -0.54050],
    [-0.54050, -0.54025],
    [-0.54025, -0.54000],
    [-0.54000, -0.53975],
    [-0.53975, -0.53950],
    [-0.53950, -0.53925],
    [-0.53925, -0.53900],
    [-0.53900, -0.53875],
    [-0.53875, -0.53850],
    [-0.53850, -0.53825],
    [-0.53825, -0.53800],
    [-0.53800, -0.53775],
    [-0.53775, -0.53750],
    [-0.53750, -0.53725],
    [-0.53725, -0.53700],
    [-0.53700, -0.53675],
    [-0.53675, -0.53650],
    [-0.53650, -0.53625],
    [-0.53625, -0.53600],
    [-0.53600, -0.53575],
    [-0.53575, -0.53550],
    [-0.53550, -0.53525],
    [-0.53525, -0.53500]
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
