#!/usr/bin/python3

import os
import subprocess
import multiprocessing

verisig_path = '../verisig'
flowstar_path = '../flowstar/flowstar'
output_path = 'output_MCC_0.56300_0.54900'
xml_path = 'MCC_600.xml'
dnn1_yaml = 'networks/Generator2x50_100_Decay1Loss1.113e-06.yml'
dnn2_yaml = 'networks/Contraster1x50_Decay1Loss57.36.yml'
dnn3_yaml = 'networks/Image2StateRobuster1x20ImageSize20x30Decay0.001.yml'
dnn4_yaml = 'networks/ControllerSigmoid16x16.yml'

if not os.path.exists(output_path):
        os.mkdir(output_path)


legend = ['X1_LOWER', 'X1_UPPER']

test_set = [
    [-0.56300, -0.56275],
    [-0.56275, -0.56250],
    [-0.56250, -0.56225],
    [-0.56225, -0.56200],
    [-0.56200, -0.56175],
    [-0.56175, -0.56150],
    [-0.56150, -0.56125],
    [-0.56125, -0.56100],
    [-0.56100, -0.56075],
    [-0.56075, -0.56050],
    [-0.56050, -0.56025],
    [-0.56025, -0.56000],
    [-0.56000, -0.55975],
    [-0.55975, -0.55950],
    [-0.55950, -0.55925],
    [-0.55925, -0.55900],
    [-0.55900, -0.55875],
    [-0.55875, -0.55850],
    [-0.55850, -0.55825],
    [-0.55825, -0.55800],
    [-0.55800, -0.55775],
    [-0.55775, -0.55750],
    [-0.55750, -0.55725],
    [-0.55725, -0.55700],
    [-0.55700, -0.55675],
    [-0.55675, -0.55650],
    [-0.55650, -0.55625],
    [-0.55625, -0.55600],
    [-0.55600, -0.55575],
    [-0.55575, -0.55550],
    [-0.55550, -0.55525],
    [-0.55525, -0.55500],
    [-0.55500, -0.55475],
    [-0.55475, -0.55450],
    [-0.55450, -0.55425],
    [-0.55425, -0.55400],
    [-0.55400, -0.55375],
    [-0.55375, -0.55350],
    [-0.55350, -0.55325],
    [-0.55325, -0.55300],
    [-0.55300, -0.55275],
    [-0.55275, -0.55250],
    [-0.55250, -0.55225],
    [-0.55225, -0.55200],
    [-0.55200, -0.55175],
    [-0.55175, -0.55150],
    [-0.55150, -0.55125],
    [-0.55125, -0.55100],
    [-0.55100, -0.55075],
    [-0.55075, -0.55050],
    [-0.55050, -0.55025],
    [-0.55025, -0.55000],
    [-0.55000, -0.54975],
    [-0.54975, -0.54950],
    [-0.54950, -0.54925],
    [-0.54925, -0.54900]
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
