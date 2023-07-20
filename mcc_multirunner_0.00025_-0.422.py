#!/usr/bin/python3

import os
import subprocess
import multiprocessing

verisig_path = '../verisig'
flowstar_path = '../flowstar/flowstar'
output_path = 'output_MCC_0.42200_0.40800'
xml_path = 'MCC_600.xml'
dnn1_yaml = 'networks/Generator2x50_100_Decay1Loss1.113e-06.yml'
dnn2_yaml = 'networks/Contraster1x50_Decay1Loss57.36.yml'
dnn3_yaml = 'networks/Image2StateRobuster1x20ImageSize20x30Decay0.001.yml'
dnn4_yaml = 'networks/ControllerSigmoid16x16.yml'

if not os.path.exists(output_path):
        os.mkdir(output_path)


legend = ['X1_LOWER', 'X1_UPPER']

test_set = [
    [-0.42200, -0.42175],
    [-0.42175, -0.42150],
    [-0.42150, -0.42125],
    [-0.42125, -0.42100],
    [-0.42000, -0.42075],
    [-0.42075, -0.42050],
    [-0.42050, -0.42025],
    [-0.42025, -0.42000],
    [-0.42000, -0.41975],
    [-0.41975, -0.41950],
    [-0.41950, -0.41925],
    [-0.41925, -0.41900],
    [-0.41900, -0.41875],
    [-0.41875, -0.41850],
    [-0.41850, -0.41825],
    [-0.41825, -0.41800],
    [-0.41800, -0.41775],
    [-0.41775, -0.41750],
    [-0.41750, -0.41725],
    [-0.41725, -0.41700],
    [-0.41700, -0.41675],
    [-0.41675, -0.41650],
    [-0.41650, -0.41625],
    [-0.41625, -0.41600],
    [-0.41600, -0.41575],
    [-0.41575, -0.41550],
    [-0.41550, -0.41525],
    [-0.41525, -0.41500],
    [-0.41500, -0.41475],
    [-0.41475, -0.41450],
    [-0.41450, -0.41425],
    [-0.41425, -0.41400],
    [-0.41400, -0.41375],
    [-0.41375, -0.41350],
    [-0.41350, -0.41325],
    [-0.41325, -0.41300],
    [-0.41300, -0.41275],
    [-0.41275, -0.41250],
    [-0.41250, -0.41225],
    [-0.41225, -0.41200],
    [-0.41200, -0.41175],
    [-0.41175, -0.41150],
    [-0.41150, -0.41125],
    [-0.41125, -0.41100],
    [-0.41100, -0.41075],
    [-0.41075, -0.41050],
    [-0.41050, -0.41025],
    [-0.41025, -0.41000],
    [-0.41000, -0.40975],
    [-0.40975, -0.40950],
    [-0.40950, -0.40925],
    [-0.40925, -0.40900],
    [-0.40900, -0.40875],
    [-0.40875, -0.40850],
    [-0.40850, -0.40825],
    [-0.40825, -0.40800]
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
