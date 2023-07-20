#!/usr/bin/python3

import os
import subprocess
import multiprocessing

verisig_path = '../verisig'
flowstar_path = '../flowstar/flowstar'
output_path = 'output_MCC_0.50700_0.50000'
xml_path = 'MCC_600.xml'
dnn1_yaml = 'networks/Generator2x50_100_Decay1Loss1.113e-06.yml'
dnn2_yaml = 'networks/Contraster1x50_Decay1Loss57.36.yml'
dnn3_yaml = 'networks/Image2StateRobuster1x20ImageSize20x30Decay0.001.yml'
dnn4_yaml = 'networks/ControllerSigmoid16x16.yml'

if not os.path.exists(output_path):
        os.mkdir(output_path)


legend = ['X1_LOWER', 'X1_UPPER']

test_set = [
    [-0.50700, -0.50675],
    [-0.50675, -0.50650],
    [-0.50650, -0.50625],
    [-0.50625, -0.50600],
    [-0.50600, -0.50575],
    [-0.50575, -0.50550],
    [-0.50550, -0.50525],
    [-0.50525, -0.50500],
    [-0.50500, -0.50475],
    [-0.50475, -0.50450],
    [-0.50450, -0.50425],
    [-0.50425, -0.50400],
    [-0.50400, -0.50375],
    [-0.50375, -0.50350],
    [-0.50350, -0.50325],
    [-0.50325, -0.50300],
    [-0.50300, -0.50275],
    [-0.50275, -0.50250],
    [-0.50250, -0.50225],
    [-0.50225, -0.50200],
    [-0.50200, -0.50175],
    [-0.50175, -0.50150],
    [-0.50150, -0.50125],
    [-0.50125, -0.50100],
    [-0.50100, -0.50075],
    [-0.50075, -0.50050],
    [-0.50050, -0.50025],
    [-0.50025, -0.50000]
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
