#!/usr/bin/python3

import os
import subprocess
import multiprocessing

verisig_path = '../verisig'
flowstar_path = '../flowstar/flowstar'
output_path = 'output_MCC_0.53500_0.52100'
xml_path = 'MCC_600.xml'
dnn1_yaml = 'networks/Generator2x50_100_Decay1Loss1.113e-06.yml'
dnn2_yaml = 'networks/Contraster1x50_Decay1Loss57.36.yml'
dnn3_yaml = 'networks/Image2StateRobuster1x20ImageSize20x30Decay0.001.yml'
dnn4_yaml = 'networks/ControllerSigmoid16x16.yml'

if not os.path.exists(output_path):
        os.mkdir(output_path)


legend = ['X1_LOWER', 'X1_UPPER']

test_set = [
    [-0.53500, -0.53475],
    [-0.53475, -0.534350],
    [-0.53450, -0.53425],
    [-0.53425, -0.53400],
    [-0.53400, -0.53375],
    [-0.53375, -0.53350],
    [-0.53350, -0.53325],
    [-0.53325, -0.53300],
    [-0.53300, -0.53275],
    [-0.53275, -0.53250],
    [-0.53250, -0.53225],
    [-0.53225, -0.53200],
    [-0.53200, -0.53175],
    [-0.53175, -0.53150],
    [-0.53150, -0.53125],
    [-0.53125, -0.53100],
    [-0.53100, -0.53075],
    [-0.53075, -0.53050],
    [-0.53050, -0.53025],
    [-0.53025, -0.53000],
    [-0.53000, -0.52975],
    [-0.52975, -0.52950],
    [-0.52950, -0.52925],
    [-0.52925, -0.52900],
    [-0.52900, -0.52875],
    [-0.52875, -0.52850],
    [-0.52850, -0.52825],
    [-0.52825, -0.52800],
    [-0.52800, -0.52775],
    [-0.52775, -0.52750],
    [-0.52750, -0.52725],
    [-0.52725, -0.52700],
    [-0.52700, -0.52675],
    [-0.52675, -0.52650],
    [-0.52650, -0.52625],
    [-0.52625, -0.52600],
    [-0.52600, -0.52575],
    [-0.52575, -0.52550],
    [-0.52550, -0.52525],
    [-0.52525, -0.52500],
    [-0.52500, -0.52475],
    [-0.52475, -0.52450],
    [-0.52450, -0.52425],
    [-0.52425, -0.52400],
    [-0.52400, -0.52375],
    [-0.52375, -0.52350],
    [-0.52350, -0.52325],
    [-0.52325, -0.52300],
    [-0.52300, -0.52275],
    [-0.52275, -0.52250],
    [-0.52250, -0.52225],
    [-0.52225, -0.52200],
    [-0.52200, -0.52175],
    [-0.52175, -0.52150],
    [-0.52150, -0.52125],
    [-0.52125, -0.52100]
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
