#!/usr/bin/python3

import os
import subprocess
import multiprocessing

def main()
    xml_path = 'MCC_600.xml'
    verisig_path = '../verisig'
    flowstar_path = '../flowstar/flowstar'
    dnn1_yaml = 'networks/Generator2x50_100_Decay1Loss1.113e-06.yml'
    dnn2_yaml = 'networks/Contraster1x50_Decay1Loss57.36.yml'
    dnn3_yaml = 'networks/Image2StateRobuster1x20ImageSize20x30Decay0.001.yml'
    dnn4_yaml = 'networks/ControllerSigmoid16x16.yml'
    args = sys.argv[1:]
    
    int_ind = args.index('-i')
    start_ind = args.index('-s')
    n_ind = args.index('-n')
    
    start_str = args[start_ind+1]
    start_val = float(start_str)
    
    int_str = args[int_ind+1]
    int_val = float(int_str)
    
    n_str = args[n_ind+1]
    n_ints = int(n_str)
    
    end_val = start_val + int_val * n_ints
    
    end_str = str(end_val)
    
    output_path = 'output_c_MCC_'+int_str+'_' + start_str + '_' + end_str

    if not os.path.exists(output_path):
        os.mkdir(output_path)


    legend = ['X1_LOWER', 'X1_UPPER']
    test_set = []
    for i in range(num_ints): 
        interval = [start_val, start_val+int_val]
        test_set.append(interval)
        start_val = start_val+int_val
    if test_set[-1][1] != end_val:
        print('Error defining intervals, ending does not match expected.')
        return -1
    else:
        print("Building the base model...")
        subprocess.run([verisig_path, '-vc=MCC_multi.yml', '-o' ,'-nf', xml_path, dnn1_yaml, dnn2_yaml, dnn3_yaml, dnn4_yaml])

        with open('MCC_600.model', 'r') as f:
            model = f.read()
        print("Starting parallel verification")
        num_parallel = num_ints
        with multiprocessing.Pool(processes=num_parallel) as pool:
            pool.map(evaluate_conditions, test_set)
        return 0



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

