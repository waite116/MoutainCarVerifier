#!/usr/bin/python3

import os
import subprocess
import multiprocessing

verisig_path = '../verisig'
flowstar_path = '../flowstar/flowstar'
output_path = 'output_MCC_0.57700_0.56300'
xml_path = 'MCC_600.xml'
dnn1_yaml = 'networks/Generator2x50_100_Decay1Loss1.113e-06.yml'
dnn2_yaml = 'networks/Contraster1x50_Decay1Loss57.36.yml'
dnn3_yaml = 'networks/Image2StateRobuster1x20ImageSize20x30Decay0.001.yml'
dnn4_yaml = 'networks/ControllerSigmoid16x16.yml'

if not os.path.exists(output_path):
        os.mkdir(output_path)


legend = ['X1_LOWER', 'X1_UPPER']

test_set = [
    [-0.57700, -0.57675],
    [-0.57675, -0.57650],
    [-0.57650, -0.57625],
    [-0.57625, -0.57600],
    [-0.57600, -0.57575],
    [-0.57575, -0.57550],
    [-0.57550, -0.57525],
    [-0.57525, -0.57500],
    [-0.57500, -0.57475],
    [-0.57475, -0.57450],
    [-0.57450, -0.57425],
    [-0.57425, -0.57400],
    [-0.57400, -0.57375],
    [-0.57375, -0.57350],
    [-0.57350, -0.57325],
    [-0.57325, -0.57300],
    [-0.57300, -0.57275],
    [-0.57275, -0.57250],
    [-0.57250, -0.57225],
    [-0.57225, -0.57200],
    [-0.57200, -0.57175],
    [-0.57175, -0.57150],
    [-0.57150, -0.57125],
    [-0.57125, -0.57100],
    [-0.57100, -0.57075],
    [-0.57075, -0.57050],
    [-0.57050, -0.57025],
    [-0.57025, -0.57000],
    [-0.57000, -0.56975],
    [-0.56975, -0.56950],
    [-0.56950, -0.56925],
    [-0.56925, -0.56900],
    [-0.56900, -0.56875],
    [-0.56875, -0.56850],
    [-0.56850, -0.56825],
    [-0.56825, -0.56800],
    [-0.56800, -0.56775],
    [-0.56775, -0.56750],
    [-0.56750, -0.56725],
    [-0.56725, -0.56700],
    [-0.56700, -0.56675],
    [-0.56675, -0.56650],
    [-0.56650, -0.56625],
    [-0.56625, -0.56600],
    [-0.56600, -0.56575],
    [-0.56575, -0.56550],
    [-0.56550, -0.56525],
    [-0.56525, -0.56500],
    [-0.56500, -0.56475],
    [-0.56475, -0.56450],
    [-0.56450, -0.56425],
    [-0.56425, -0.56400],
    [-0.56400, -0.56375],
    [-0.56375, -0.56350],
    [-0.56350, -0.56325],
    [-0.56325, -0.56300]
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
