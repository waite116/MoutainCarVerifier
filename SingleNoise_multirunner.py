#!/usr/bin/python3
import sys
import os
import subprocess
import multiprocessing
import numpy as np
from decimal import *
from functools import partial
def main():
    xml_path = 'SingleNoiseModel.xml'
    verisig_path = '../verisig'
    flowstar_path = '../flowstar/flowstar'
    dnn1_yaml = 'networks/Generator2x50_100_Decay1TestLoss1.035e-06.yml'
    dnn3_yaml = 'networks/CompositeRegressor3x20_20_20_Decay0.001TestLoss5.15e-06Composite.yml'
    dnn4_yaml = 'networks/ControllerSigmoid16x16.yml'
    args = sys.argv[1:]
    
    int_ind = args.index('-i')
    start_ind = args.index('-s')
    n_ind = args.index('-n')
    noise_type_ind = args.index('-t')
    noise_type_str = args[noise_type_ind + 1].lower()
    if noise_type_str == 'blur':
        dnn2_yaml = 'networks/Blurer1x50_Decay10TestLoss47.28.yml'
        model_yaml = 'SingleNoiseBlur_multi.yml'
    elif noise_type_str == 'contrast':
        dnn2_yaml = 'networks/Contraster1x50_Decay5TestLoss47.74.yml'
        model_yaml = 'SingleNoiseContrast_multi.yml'
    else:
        print('Specify noise type with a -t parameter, e.g. -t blur')
        return

    start_str = args[start_ind+1]
    start_val = Decimal(start_str)
    print(start_val)
    int_str = args[int_ind+1]
    int_val = Decimal(int_str)
    print(int_val)
    
    n_str = args[n_ind+1]
    n_ints = int(n_str)
    
    end_val = start_val + int_val * n_ints
    
    end_str = str(end_val)
    
    output_path = 'output_single_noise_'+ noise_type_str+ '_' +int_str+'_' + start_str + '_' + end_str

    if not os.path.exists(output_path):
        os.mkdir(output_path)


    legend = ['X1_LOWER', 'X1_UPPER']
    
    test_set = []
    for i in range(n_ints): 
        interval = [start_val, start_val+int_val]
        test_set.append(interval)
        start_val = start_val+int_val
    print(test_set)
    if test_set[-1][1] != end_val:
        print('Error defining intervals, ending does not match expected.')
        return -1
    else:
        print("Building the base model...")
        subprocess.run([verisig_path, '-vc='+model_yaml, '-o' ,'-nf', xml_path, dnn1_yaml, dnn2_yaml, dnn3_yaml, dnn4_yaml])

        with open('SingleNoiseModel.model', 'r') as f:
            in_model = f.read()
        print("Starting parallel verification")
        num_parallel = n_ints
        with multiprocessing.Pool(processes=num_parallel) as pool:
            pool.map(partial(evaluate_conditions, legend=legend, model=in_model, output_path=output_path, flowstar_path=flowstar_path, dnn1_yaml=dnn1_yaml, dnn2_yaml=dnn2_yaml, dnn3_yaml=dnn3_yaml, dnn4_yaml=dnn4_yaml), test_set)
        return 0



#===========================================================================================
# Begin Parallel Function
#==========================================================================================

def evaluate_conditions(conditions, legend, model, output_path, flowstar_path, dnn1_yaml, dnn2_yaml, dnn3_yaml, dnn4_yaml):
    test_model = model
    for i in range(len(legend)):
        test_model = test_model.replace(legend[i], str(conditions[i]))

    with open(output_path + '/' + str(conditions[0]) + ', ' + str(conditions[1]) + '.txt', 'w') as f:
        subprocess.run(flowstar_path + ' ' + dnn1_yaml + ' ' + dnn2_yaml + ' ' + dnn3_yaml + ' ' + dnn4_yaml, input=test_model, shell=True, universal_newlines=True, stdout=f)
    print('Finished Interval: [' + str(conditions[0]) + ', ' + str(conditions[1]) + ']')
#===========================================================================================
# End Parallel Function
#===========================================================================================

if __name__ == '__main__':
    main()
