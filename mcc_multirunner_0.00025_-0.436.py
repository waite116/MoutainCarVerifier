#!/usr/bin/python3

import os
import subprocess
import multiprocessing

verisig_path = '../verisig'
flowstar_path = '../flowstar/flowstar'
output_path = 'output_MCC_0.43600_0.42200'
xml_path = 'MCC_600.xml'
dnn1_yaml = 'networks/Generator2x50_100_Decay1Loss1.113e-06.yml'
dnn2_yaml = 'networks/Contraster1x50_Decay1Loss57.36.yml'
dnn3_yaml = 'networks/Image2StateRobuster1x20ImageSize20x30Decay0.001.yml'
dnn4_yaml = 'networks/ControllerSigmoid16x16.yml'

if not os.path.exists(output_path):
        os.mkdir(output_path)


legend = ['X1_LOWER', 'X1_UPPER']

test_set = [
    [-0.43600, -0.43575],
    [-0.43575, -0.43550],
    [-0.43550, -0.43525],
    [-0.43525, -0.43500],
    [-0.43500, -0.43475],
    [-0.43475, -0.43450],
    [-0.43450, -0.43425],
    [-0.43425, -0.43400],
    [-0.43400, -0.43375],
    [-0.43375, -0.43350],
    [-0.43350, -0.43325],
    [-0.43325, -0.43300],
    [-0.43300, -0.43275],
    [-0.43275, -0.43250],
    [-0.43250, -0.43225],
    [-0.43225, -0.43200],
    [-0.43200, -0.43175],
    [-0.43175, -0.43150],
    [-0.43150, -0.43125],
    [-0.43125, -0.43100],
    [-0.43100, -0.43075],
    [-0.43075, -0.43050],
    [-0.43050, -0.43025],
    [-0.43025, -0.43000],
    [-0.43000, -0.42975],
    [-0.42975, -0.42950],
    [-0.42950, -0.42925],
    [-0.42925, -0.42900],
    [-0.42900, -0.42875],
    [-0.42875, -0.42850],
    [-0.42850, -0.42825],
    [-0.42825, -0.42800],
    [-0.42800, -0.42775],
    [-0.42775, -0.42750],
    [-0.42750, -0.42725],
    [-0.42725, -0.42700],
    [-0.42700, -0.42675],
    [-0.42675, -0.42650],
    [-0.42650, -0.42625],
    [-0.42625, -0.42600],
    [-0.42600, -0.42575],
    [-0.42575, -0.42550],
    [-0.42550, -0.42525],
    [-0.42525, -0.42500],
    [-0.42500, -0.42475],
    [-0.42475, -0.42450],
    [-0.42450, -0.42425],
    [-0.42425, -0.42400],
    [-0.42400, -0.42375],
    [-0.42375, -0.42350],
    [-0.42350, -0.42325],
    [-0.42325, -0.42300],
    [-0.42300, -0.42275],
    [-0.42275, -0.42250],
    [-0.42250, -0.42225],
    [-0.42225, -0.42200]
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
