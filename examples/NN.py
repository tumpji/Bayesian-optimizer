#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: NN.py
#  DESCRIPTION: Optimize NN model using GPyOpt
#        USAGE: run as a python3.6 script 
# REQUIREMENTS: numpy, sklearn, boto3, optimizeTools (provided library)
#               You may need NeuralNetwork class, but it is not nesses
#
#      LICENCE:
#
#         BUGS:
#        NOTES:
#       AUTHOR: Jiří Tumpach (tumpji),
# ORGANIZATION:
#      VERSION: 1.0
#      CREATED: 2019 03.25.
# =============================================================================
#import pdb; pdb.set_trace()
import numpy as np

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

import os
os.nice(200)

from optimizeTools import *
from dataloader import DatasetGenerator

import time, datetime

from NeuralNetwork import NeuralNetworkClassifier












################################################################################
######################## Options ###############################################
################################################################################

Categorical('weeks', [1])



layersoo = Layers('layer', 4, 400)
Categorical('activation', NeuralNetworkClassifier.ACTIVATIONS)
Continuous('alpha', 0., 1.)

Continuous('learning_rate', 0.0001, 0.01) 
Continuous('learning_rate_decay', 0.0001, 0.05)

Continuous('l1_reg', 0, 0.1)
Continuous('l2_reg', 0, 0.1)
#Discrete('use_l1', [False, True]) #Discrete('use_l2', [False, True])

Continuous('dropout', 0, 0.7)
Categorical('use_dropout', [False, True])

Continuous('gaussian_noise', 0, 1)
Categorical('use_gaussian_noise', [False, True])
Categorical('use_batchnorm', [False, True])
Categorical('use_amsgrad', [False, True])

#Continuous('batchnorm_momentum', 0.9, 0.999)

DiscreteInt('batch_size', range(10,1000,10))

CategoricalLabel('norma_feature',
    [
    #('I', Identity), 
    #('MinMax', MinMaxScaler),
    #('MaxAbs', MaxAbsScaler), 
    ('Standard', StandardScaler),
    ('Robust', RobustScaler),
    ('MinMax', lambda: MinMaxScaler((-0.5,0.5))),
    ], default_index=0)





################################################################################
######################## Algorithm #############################################
################################################################################


# optimization function
def NN_test_function(vals):
    s = GpyOptOption.convert_to_objects(*vals[0])
    dbd = GpyOptOption.convert_to_db_description(*vals[0])
    dbr = " ".join(GpyOptOption.tagit(vals[0]))

    print('Started: {} {}'.format(datetime.datetime.now().strftime("%D %H:%M:%S"), dbr))
    start_time = datetime.datetime.now()

    # algorithm
    version = (1,1,0)
    layers = layersoo.get_layers_from_object_dict(s)
    batch_size = s['batch_size']
    l1 = s['l1_reg'] 
    l2 = s['l2_reg']
    dropout = s['dropout'] if s['use_dropout'] else 0.
    gaussian_noise_stddev = s['gaussian_noise'] if s['use_gaussian_noise'] else 0.
    batchnormalization = s['use_batchnorm']

    norma_feature = s['norma_feature']()

    ds = DatasetGenerator(150000, train_iterations=1, train=True)
    network = NeuralNetworkClassifier(
            input_dim=540, output_dim=2, 
            layers=layers, activation=s['activation'], batch_size=batch_size,
            learning_rate=s['learning_rate'], learning_rate_decay=s['learning_rate_decay'], 
            l1=l1, l2=l2, dropout=dropout, gaussian_noise_stddev=gaussian_noise_stddev, 
            batchnormalization=batchnormalization,
            use_amsgrad=s['use_amsgrad'],
            alpha=s['alpha'], verbose=1)

    acc_final, tested = 0., 0.

    for (Xqueue, Yqueue), (Xweek, Yweek) in ds.queue_iterator():
        Xqueue = norma_feature.fit_transform(Xqueue)
        Xweek = norma_feature.transform(Xweek)

        network.fit(Xqueue, Yqueue) 
        acc = network.score(Xweek, Yweek)[1]

        acc_final += acc
        tested += 1.
        print("Act: {}".format(acc))
    acc_final /= tested

    end_time = (datetime.datetime.now() - start_time).seconds
    write_kwargs_to_db(TABLE='NeuralNetwork', VERSION=version, RESULT=acc_final, 
            DURATION=end_time, **dbd)
    print('Ended with {} ({:.5}min): {}'.format(acc_final, end_time/60, dbr))
    return -acc_final





################################################################################
######################## Optimization ##########################################
################################################################################

def load_my_results():
    if True:
        f = []
        f.append(lambda x: x['VERSION'] >= (1,1,0))
        db = read_all(TABLE='NeuralNetwork', FILTER=tuple(f))

        a = GpyOptOption.convert_from_previous_results(db, maximize=True)
        if len(a):
            print("Loaded {} samples...".format(a['Y'].shape[0]))
            return a
    print("Loaded 0 samples...")
    return None
    

def define_problem(first_time, main_optimizer=False, cores=1):
    domain = GpyOptOption.generate_domains() if define_problem.domain is None else define_problem.domain
    constraints = GpyOptOption.generate_constraints() if define_problem.constraints is None else define_problem.constraints

    #evaluator_type='thompson_sampling'
    #evaluator_type='sequential'
    #evaluator_type = 'random'
    EVALUATOR_TYPE = 'local_penalization'

    INITIAL_DESIGN_NUMDATA = cores
    BATCH_SIZE = cores
    NUM_CORES = cores
    res = load_my_results()

    if main_optimizer:
        BATCH_SIZE = 1
        NUM_CORES = 1
        INITIAL_DESIGN_NUMDATA = 1
        EVALUATOR_TYPE = 'sequential'

    res = res if res is not None else {}
    # TODO: change to seq + local_penalization based on

    myProblem = GPyOpt.methods.BayesianOptimization(
            NN_test_function,
            domain=domain, constraints=constraints, evaluator_type=EVALUATOR_TYPE,
            **res,
            #de_duplication=True,
            initial_design_numdata=(INITIAL_DESIGN_NUMDATA if first_time else 0),
            batch_size=BATCH_SIZE, num_cores=NUM_CORES,
            verbosity=True, verbosity_model=True)
    return myProblem

define_problem.domain = None
define_problem.constraints = None


if __name__ == '__main__':
    # optimization part
    import GPy
    import GPyOpt
    import argparse

    parser = argparse.ArgumentParser(description='Optimize some function')
    parser.add_argument('--main', action='store_true')
    parser.add_argument('--first', action='store_true')
    args = parser.parse_args()

    if args.main:
        print("MAIN OPTIMIZER:")


    # formating stuff
    import colorama
    from colorama import Fore, Back, Style
    colorama.init()
    import textwrap
    tw = textwrap.TextWrapper(
            initial_indent="\t\t",
            subsequent_indent="\t\t",
            break_long_words=False, width=70)


    MAX_NON_DECRESING_ITERATIONS = 20
    MAX_ITERATIONS = None
    MAX_TIME_M = None

    GpyOptOption.finalize()

    iteration = 0
    best_result = np.inf
    countdown = MAX_NON_DECRESING_ITERATIONS

    start_time = datetime.datetime.now()
    iteration_started = start_time
    myProblem = define_problem(args.first, main_optimizer=args.main)

    while countdown > 0 and \
        (MAX_ITERATIONS is None or iteration < MAX_ITERATIONS) and \
        (MAX_TIME_M is None or (datetime.datetime() - start_time).seconds < MAX_TIME_M*60):

        now = datetime.datetime.now()

        if hasattr(myProblem, 'fx_opt'):
            if best_result > myProblem.fx_opt:
                best_result = myProblem.fx_opt
                countdown = MAX_NON_DECRESING_ITERATIONS
            else:
                countdown -= 1

            print(("\nGpyOpt iteration {}: date: {} from start: {} iteration: {}\n" +
                  "\tExpectation: {} at\n{}\n").format( 
                Style.BRIGHT + str(iteration) + Style.RESET_ALL, 
                Style.BRIGHT + now.strftime("%D %H:%M:%S") + Style.RESET_ALL,
                Style.BRIGHT + (str(now - start_time).split('.')[0]) + Style.RESET_ALL,
                Style.BRIGHT + (str(now - iteration_started).split('.')[0]) + Style.RESET_ALL,
                Style.BRIGHT + Fore.GREEN + "{:.5}".format(float(myProblem.fx_opt)) + Style.RESET_ALL, 
                "\n".join(x.replace("=", " = ") for x in
                    tw.wrap("   ".join(GpyOptOption.tagit(myProblem.x_opt))))
                ))

        # optimization iteration
        iteration += 1
        iteration_started = now
        myProblem = define_problem(False, main_optimizer=args.main)
        myProblem.run_optimization(1)





        # myProblem = ... TODO with load from db












