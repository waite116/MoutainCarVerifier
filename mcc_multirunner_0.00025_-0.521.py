#!/usr/bin/python3

import os
import subprocess
import multiprocessing

verisig_path = '../verisig'
flowstar_path = '../flowstar/flowstar'
output_path = 'output_MCC_0.52100_0.50700'
xml_path = 'MCC_600.xml'
dnn1_yaml = 'networks/Generator2x50_100_Decay1Loss1.113e-06.yml'
dnn2_yaml = 'networks/Contraster1x50_Decay1Loss57.36.yml'
dnn3_yaml = 'networks/Image2StateRobuster1x20ImageSize20x30Decay0.001.yml'
dnn4_yaml = 'networks/ControllerSigmoid16x16.yml'

if not os.path.exists(output_path):
        os.mkdir(output_path)


legend = ['X1_LOWER', 'X1_UPPER']

test_set = [
    [-0.52100, -0.52075],
    [-0.52075, -0.52050],
    [-0.52050, -0.52025],
    [-0.52025, -0.52000],
    [-0.52000, -0.51975],
    [-0.51975, -0.51950],
    [-0.51950, -0.51925],
    [-0.51925, -0.51900],
    [-0.51900, -0.51875],
    [-0.51875, -0.51850],
    [-0.51850, -0.51825],
    [-0.51825, -0.51800],
    [-0.51800, -0.51775],
    [-0.51775, -0.51750],
    [-0.51750, -0.51725],
    [-0.51725, -0.51700],
    [-0.51700, -0.51675],
    [-0.51675, -0.51650],
    [-0.51650, -0.51625],
    [-0.51625, -0.51600],
    [-0.51600, -0.51575],
    [-0.51575, -0.51550],
    [-0.51550, -0.51525],
    [-0.51525, -0.51500],
    [-0.51500, -0.51475],
    [-0.51475, -0.51450],
    [-0.51450, -0.51425],
    [-0.51425, -0.51400],
    [-0.51400, -0.51375],
    [-0.51375, -0.51350],
    [-0.51350, -0.51325],
    [-0.51325, -0.51300],
    [-0.51300, -0.51275],
    [-0.51275, -0.51250],
    [-0.51250, -0.51225],
    [-0.51225, -0.51200],
    [-0.51200, -0.51175],
    [-0.51175, -0.51150],
    [-0.51150, -0.51125],
    [-0.51125, -0.51100],
    [-0.51100, -0.51075],
    [-0.51075, -0.51050],
    [-0.51050, -0.51025],
    [-0.51025, -0.51000],
    [-0.51000, -0.50975],
    [-0.50975, -0.50950],
    [-0.50950, -0.50925],
    [-0.50925, -0.50900],
    [-0.50900, -0.50875],
    [-0.50875, -0.50850],
    [-0.50850, -0.50825],
    [-0.50825, -0.50800],
    [-0.50800, -0.50775],
    [-0.50775, -0.50750],
    [-0.50750, -0.50725],
    [-0.50725, -0.50700]
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
