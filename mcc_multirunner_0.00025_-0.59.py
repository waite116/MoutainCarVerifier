#!/usr/bin/python3

import os
import subprocess
import multiprocessing

verisig_path = '../verisig'
flowstar_path = '../flowstar/flowstar'
output_path = 'output_MCC_0.59000_0.57700'
xml_path = 'MCC_600.xml'
dnn1_yaml = 'networks/Generator2x50_100_Decay1Loss1.113e-06.yml'
dnn2_yaml = 'networks/Contraster1x50_Decay1Loss57.36.yml'
dnn3_yaml = 'networks/Image2StateRobuster1x20ImageSize20x30Decay0.001.yml'
dnn4_yaml = 'networks/ControllerSigmoid16x16.yml'

if not os.path.exists(output_path):
        os.mkdir(output_path)


legend = ['X1_LOWER', 'X1_UPPER']

test_set = [
    [-0.59000, -0.58975],
    [-0.58975, -0.58950],
    [-0.58950, -0.58925],
    [-0.58925, -0.58900],
    [-0.58900, -0.58875],
    [-0.58875, -0.58850],
    [-0.58850, -0.58825],
    [-0.58825, -0.58800],
    [-0.58800, -0.58775],
    [-0.58775, -0.58750],
    [-0.58750, -0.58725],
    [-0.58725, -0.58700],
    [-0.58700, -0.58675],
    [-0.58675, -0.58650],
    [-0.58650, -0.58625],
    [-0.58625, -0.58600],
    [-0.58600, -0.58575],
    [-0.58575, -0.58550],
    [-0.58550, -0.58525],
    [-0.58525, -0.58500],
    [-0.58500, -0.58475],
    [-0.58475, -0.58450],
    [-0.58450, -0.58425],
    [-0.58425, -0.58400],
    [-0.58400, -0.58375],
    [-0.58375, -0.58350],
    [-0.58350, -0.58325],
    [-0.58325, -0.58300],
    [-0.58300, -0.58275],
    [-0.58275, -0.58250],
    [-0.58250, -0.58225],
    [-0.58225, -0.58200],
    [-0.58200, -0.58175],
    [-0.58175, -0.58150],
    [-0.58150, -0.58125],
    [-0.58125, -0.58100],
    [-0.58100, -0.58075],
    [-0.58075, -0.58050],
    [-0.58050, -0.58025],
    [-0.58025, -0.58000],
    [-0.58000, -0.57975],
    [-0.57975, -0.57950],
    [-0.57950, -0.57925],
    [-0.57925, -0.57900],
    [-0.57900, -0.57875],
    [-0.57875, -0.57850],
    [-0.57850, -0.57825],
    [-0.57825, -0.57800],
    [-0.57800, -0.57775],
    [-0.57775, -0.57750],
    [-0.57750, -0.57725],
    [-0.57725, -0.57700]
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
