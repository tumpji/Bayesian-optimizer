#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: kNN.py
#  DESCRIPTION: Optimize kNN model using GPyOpt
#        USAGE: run as a python3.6 script 
# REQUIREMENTS: numpy, sklearn, boto3 (optimizeTools - provided module)
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
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

import os
os.nice(200)

from optimizeTools import *

# you shoud have your own way of providing data to algorithm
from dataloader import DatasetGenerator

import time, datetime












################################################################################
######################## Options ###############################################
################################################################################

class Identity(sklearn.base.TransformerMixin):
    """ do nothing """
    def __init__(self): super(Identity, self).__init__()
    def fit(self, X): return self
    def transform(self, X): return X


CategoricalLabel('norma_feature',
    [
    #('I', Identity), 
    #('MinMax', MinMaxScaler),
    #('MaxAbs', MaxAbsScaler), 
    ('Standard', StandardScaler),
    ('Robust', RobustScaler),
    ('MinMaxNeg1', lambda: MinMaxScaler((-1,1))),
    ('MinMaxNegh', lambda: MinMaxScaler((-0.5,0.5))),
    #('Auto', Identity),
    ], default_index=0)

CategoricalLabel('norma_data',
    [
    ('I', lambda x: x),
    #('l1', lambda x : normalize(x, norm='l1')),
    #('l2', lambda x: normalize(x, norm='l2')),
    #('max', lambda x: normalize(x, norm='max')),
    ]) 

#Categorical('training_iterations', [2])
#Categorical('kernel', ['linear'])
#DiscreteInt('degree', [0])
Categorical('dual', [False])

Continuous('penalty', 0.001, 80.)

a = Categorical('penalty_type', ['l1', 'l2'])
b = Categorical('loss', ['squared_hinge'])

#a.force_values_to_others_if_equal_to(['l2'], b, ['squared_hinge'])
#a.force_values_to_others_if_equal_to(['l1'], b, ['squared_hinge'])








################################################################################
######################## Algorithm #############################################
################################################################################


# optimization function
def SVM_test_function(vals):
    s = GpyOptOption.convert_to_objects(*vals[0])
    dbd = GpyOptOption.convert_to_db_description(*vals[0])
    dbr = " ".join(GpyOptOption.tagit(vals[0]))

    print('Started: {} {}'.format(datetime.datetime.now().strftime("%D %H:%M:%S"), dbr))

    # algorithm
    version = (1,2,0)
    norma_data, norma_feature = s['norma_data'], s['norma_feature']
    norma_feature = norma_feature()

    ds = DatasetGenerator(150000, train_iterations=1, train=True)
    alg = LinearSVC(
            penalty=s['penalty_type'],
            C=s['penalty'], 
            dual=s['dual'],
            loss=s['loss']
            )

    acc_final, tested = 0., 0.

    for (Xqueue, Yqueue), (Xweek, Yweek) in ds.queue_iterator():
        Xqueue = norma_data(norma_feature.fit_transform(Xqueue))
        Xweek = norma_data(norma_feature.transform(Xweek))

        alg.fit(Xqueue, Yqueue) 
        acc = alg.score(Xweek, Yweek)

        acc_final += acc
        tested += 1.
        print("Act: {}".format(acc))

    acc_final /= tested

    write_kwargs_to_db(TABLE='SVM',
            VERSION=version,
            RESULT=acc_final,
            **dbd,
            ) # dataset info
    
    print('Ended with {}: {}'.format(acc_final, dbr))
    return -acc_final





################################################################################
######################## Optimization ##########################################
################################################################################

def load_my_results():
    if True:
        f = []
        f.append(lambda x: x['VERSION'] >= (1,2,0))
        db = read_all(TABLE='SVM', FILTER=tuple(f))

        a = GpyOptOption.convert_from_previous_results(db, maximize=True)
        if len(a):
            print("Loaded {} samples...".format(a['Y'].shape[0]))
            return a
    print("Loaded 0 samples...")
    return None
    

def define_problem(first_time, main_optimizer=False):
    domain = GpyOptOption.generate_domains() if define_problem.domain is None else define_problem.domain
    constraints = GpyOptOption.generate_constraints() if define_problem.constraints is None else define_problem.constraints

    #evaluator_type='thompson_sampling'
    #evaluator_type='sequential'
    #evaluator_type = 'random'
    EVALUATOR_TYPE = 'local_penalization'

    INITIAL_DESIGN_NUMDATA = 1
    BATCH_SIZE = 1
    NUM_CORES = 1
    res = load_my_results()

    if main_optimizer:
        BATCH_SIZE = 1
        NUM_CORES = 1
        INITIAL_DESIGN_NUMDATA = 1
        EVALUATOR_TYPE = 'sequential'

    res = res if res is not None else {}
    # TODO: change to seq + local_penalization based on

    myProblem = GPyOpt.methods.BayesianOptimization(
            SVM_test_function,
            domain=domain, constraints=constraints,
            evaluator_type=EVALUATOR_TYPE,
            initial_design_numdata=(INITIAL_DESIGN_NUMDATA if first_time else 0),
            **res,

            #de_duplication=True,
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












