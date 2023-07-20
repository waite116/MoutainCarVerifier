#!/usr/bin/python3

import os
import subprocess
import multiprocessing

verisig_path = '../verisig'
flowstar_path = '../flowstar/flowstar'
output_path = 'output_MCC_0.47500_0.46200'
xml_path = 'MCC_600.xml'
dnn1_yaml = 'networks/Generator2x50_100_Decay1Loss1.113e-06.yml'
dnn2_yaml = 'networks/Contraster1x50_Decay1Loss57.36.yml'
dnn3_yaml = 'networks/Image2StateRobuster1x20ImageSize20x30Decay0.001.yml'
dnn4_yaml = 'networks/ControllerSigmoid16x16.yml'

if not os.path.exists(output_path):
        os.mkdir(output_path)


legend = ['X1_LOWER', 'X1_UPPER']

test_set = [
    [-0.47500, -0.47475],
    [-0.47475, -0.47450],
    [-0.47450, -0.47425],
    [-0.47425, -0.47400],
    [-0.47400, -0.47375],
    [-0.47375, -0.47350],
    [-0.47350, -0.47325],
    [-0.47325, -0.47300],
    [-0.47300, -0.47275],
    [-0.47275, -0.47250],
    [-0.47250, -0.47225],
    [-0.47225, -0.47200],
    [-0.47200, -0.47175],
    [-0.47175, -0.47150],
    [-0.47150, -0.47125],
    [-0.47125, -0.47100],
    [-0.47100, -0.47075],
    [-0.47075, -0.47050],
    [-0.47050, -0.47025],
    [-0.47025, -0.47000],
    [-0.47000, -0.46975],
    [-0.46975, -0.46950],
    [-0.46950, -0.46925],
    [-0.46925, -0.46900],
    [-0.46900, -0.46875],
    [-0.46875, -0.46850],
    [-0.46850, -0.46825],
    [-0.46825, -0.46800],
    [-0.46800, -0.46775],
    [-0.46775, -0.46750],
    [-0.46750, -0.46725],
    [-0.46725, -0.46700],
    [-0.46700, -0.46675],
    [-0.46675, -0.46650],
    [-0.46650, -0.46625],
    [-0.46625, -0.46600],
    [-0.46600, -0.46575],
    [-0.46575, -0.46550],
    [-0.46550, -0.46525],
    [-0.46525, -0.46500],
    [-0.46500, -0.46475],
    [-0.46475, -0.46450],
    [-0.46450, -0.46425],
    [-0.46425, -0.46400],
    [-0.46400, -0.46375],
    [-0.46375, -0.46350],
    [-0.46350, -0.46325],
    [-0.46325, -0.46300],
    [-0.46300, -0.46275],
    [-0.46275, -0.46250],
    [-0.46250, -0.46225],
    [-0.46225, -0.46200]
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
