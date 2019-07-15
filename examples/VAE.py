#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: kNN.py
#  DESCRIPTION: Optimize kNN model using GPyOpt
#        USAGE: run as a python3.6 script 
# REQUIREMENTS: numpy, sklearn, boto3
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

import os
os.nice(200)

from optimizeTools import *
from dataloader import DatasetGenerator

import time, datetime

#from NeuralNetwork import NeuralNetworkClassifier
from NeuralAutoencoder import VAE


################################################################################
######################## Options ###############################################
################################################################################


class Identity(sklearn.base.TransformerMixin):
    """ do nothing """
    def __init__(self): super(Identity, self).__init__()
    def fit(self, X): return self
    def transform(self, X): return X
    def inverse_trainsform(self, X): return X

######## NETWORK #############
layersoo = Layers('layer', 3, 500+1)#, behaviour='no space')
DiscreteInt('latent_size', range(1, 71))

Categorical('activation', VAE.ACTIVATIONS)
Continuous('alpha', 0., 1.)

Continuous('learning_rate', 0.0005, 0.005) 
Continuous('learning_rate_decay', 0.0, 0.05)

Continuous('l2_reg', 0, 0.1)

DiscreteInt('batch_size', range(500,10000,500))

Categorical('use_batchnorm', [True])
#Categorical('use_amsgrad', [False, True])
Categorical('use_amsgrad', [True])

######### ALGORITHM ##############

CategoricalLabel('norma_feature',
    [
#    ('I', Identity), 
#    ('Standard', StandardScaler),
 #   ('Robust', RobustScaler),
    ('MinMaxH', lambda: MinMaxScaler((-0.5,0.5))),
#    ('MinMax', lambda: MinMaxScaler()),
    ], default_index=0)


Categorical('used_weeks', [2])

#Categorical('weeks_skipped', [3])
#DiscreteInt('generate_to_new_week', range(1000,300000,1000))



#Continuous('dropout', 0, 0.7)
#Categorical('use_dropout', [False, True])
#Continuous('gaussian_noise', 0, 1)
#Categorical('use_gaussian_noise', [False, True])
#Continuous('batchnorm_momentum', 0.9, 0.999)






################################################################################
######################## Algorithm #############################################
################################################################################


# optimization function
def VAE_test_function(vals):
    s = GpyOptOption.convert_to_objects(*vals[0])
    dbd = GpyOptOption.convert_to_db_description(*vals[0])
    dbr = " ".join(GpyOptOption.tagit(vals[0]))

    print('Started: {} {}'.format(datetime.datetime.now().strftime("%D %H:%M:%S"), dbr))
    start_time = datetime.datetime.now()

    # algorithm
    version = (1,4,0)

    layers = layersoo.get_layers_from_object_dict(s)
    layers.append(s['latent_size'])

    batch_size = s['batch_size']
    l2 = s['l2_reg']
    batchnormalization = s['use_batchnorm']
    norma_feature = s['norma_feature']()

    def get_network ():
        return VAE(s['loss'], input_dim=540, layers=layers, activation=s['activation'], batch_size=batch_size,
            learning_rate=s['learning_rate'], learning_rate_decay=s['learning_rate_decay'], 
            l2=l2, batchnormalization=batchnormalization,
            use_amsgrad=s['use_amsgrad'], verbose=1,
            )
        pass

    ds = DatasetGenerator(150000, train_iterations=1, train=True)
    it = ds.chunk_iterator().__iter__()

    prvni, _ = next(it)
    next(it)
    druhy, _ = next(it)
    data = norma_feature.fit_transform(np.concatenate([prvni, druhy], axis=0))
    prvni = druhy = None

    train, test = train_test_split(data, test_size=0.2, random_state=123)
    data = None

    for i in range(6):
        network1 = get_network()
        network1.fit(train, maximum_epochs=1000, limit_loss=[(2,5000.),(4,1000.),(8,250.)])
        acc = network1.score(test)

        if np.isnan(acc):
            print("Restarting training (NAN)...")
            acc = 1000.
            continue
        break

    end_time = (datetime.datetime.now() - start_time).seconds
    write_kwargs_to_db(TABLE='VAE2', VERSION=version, RESULT=acc, DURATION=end_time, **dbd)
    print('Ended with {} ({:.5}min): {}'.format(acc, end_time/60, dbr))
    return acc





################################################################################
######################## Optimization ##########################################
################################################################################

def load_my_results():
    if True:
        '''
        def convert_from_1_0_0_to_1_1_0(x):
            if x['VERSION'] == (1,0,0):
                x['latent_size'] = x['layer_2']
                x['layer_2'] = 0
            return x
        '''

        f = []
        f.append(lambda x: x['VERSION'] >= (1,4,0))
        
        db = read_all(TABLE='VAE2', FILTER=tuple(f))#, MAP=(convert_from_1_0_0_to_1_1_0,))

        a = GpyOptOption.convert_from_previous_results(db)
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
            VAE_test_function,
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
    parser.add_argument('--nothing', type=int)
    parser.add_argument('--loss', type=str, required=True)
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

    Categorical('loss', [args.loss])




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

