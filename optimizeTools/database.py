#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: database.py
#  DESCRIPTION: Manages connectin to dynamodb (AWS) database and store all key-value pairs
#               provided to write_kwargs_to_db(**kwargs) function (with TIME: datetime.datetime), 
#               you shoud  manualy provide VERSION keyword too (tuple of ints)
#        USAGE: 
#               Function 'read_all' loads all results
#                   It has 3 parameters: 
#                       TABLE: name of table inside dynamodb
#                       MAP: tuple of functions that sequentialy maps results into other 
#                       FILTER: tuple of functions that filters results True=pass
#               Function 'write_kwargs_to_db' saves provided result
#                   If has 2 parameters:
#                       TABLE: name of databaze
#                       *kwargs: all parameters to save
#                           RESULT, VERSION are required, TIME is automaticly added 
#                           VERSION shoud be tuple of integers
#      EXAMPLE read_all:
#           f = lambda x: x['VERSION'] >= (1,0,0)
#           g = lambda x: x['activation'] == 'relu'
#           def j(x):
#               if x['VERSION'] == (1,1,0):
#                   x['latent_size'] = x['layers'][-1]
#                   x['layers'] = x['layers'][:-1]
#               return x
#           db = read_all(TABLE='MyOptimizationDB', FILTER=(f,g), MAP=(j,))
#       EXAMPLE write_kwargs_to_db:
#           write_kwargs_to_db(TABLE='MyOptimizationDB', VERSION=(1,1,4), **actual_options)
#
# REQUIREMENTS: 
#           boto3 with some ~/.aws/config & ~/.aws/credentials files set up
#
#      LICENCE: 
#           This file is part of Bayesian-optimizer .

#           Bayesian-optimizer is free software: you can redistribute it and/or modify
#           it under the terms of the GNU General Public License as published by
#           the Free Software Foundation, either version 3 of the License, or
#           (at your option) any later version.

#           Bayesian-optimizer is distributed in the hope that it will be useful,
#           but WITHOUT ANY WARRANTY; without even the implied warranty of
#           MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#           GNU General Public License for more details.

#           You should have received a copy of the GNU General Public License
#           along with Bayesian-optimizer.  If not, see <https://www.gnu.org/licenses/>.
#
#         BUGS: 
#        NOTES:
#       AUTHOR: Jiří Tumpach (tumpji),
# ORGANIZATION:
#      VERSION: 1.0
#      CREATED: 2019 03.28.
# =============================================================================

import boto3
import datetime
import decimal

import random
import string
import functools

def _random_string_generator(length):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def _change_format(something, key):
    if isinstance(something, float):
        return decimal.Decimal(str(something))
    elif isinstance(something, datetime.datetime):
        return something.strftime("%Y %m.%d. %H:%M:%S")
    elif key == 'VERSION':
        return '.'.join((str(a) for a in something))
    else:
        return something

def _change_format_back(something, key):
    if isinstance(something, decimal.Decimal):
        return float(something)
    elif key == 'TIME':
        return datetime.datetime.strptime(something, "%Y %m.%d. %H:%M:%S")
    elif key == 'VERSION':
        return tuple((int(x) for x in something.split('.')))
    else:
        return something

def write_kwargs_to_db(TABLE, **kwargs):
    """
    Writes all kwargs to database (Dynamodb)
    TIME is added automaticaly
    VERSION shoud be tuple of integers
    RESULT shoud be float
    all other kwargs is not checked but shoud be (Float,Int,None,String,Boolean)
    """
    assert 'TIME' not in kwargs
    assert 'RESULT' in kwargs
    assert isinstance(kwargs['RESULT'], float)
    assert 'VERSION' in kwargs 
    assert isinstance(kwargs['VERSION'], tuple)
    assert all((isinstance(x, int) for x in kwargs['VERSION']))

    to_save = {k:_change_format(v,k) for k,v in kwargs.items()}
    time = datetime.datetime.now()#.strftime("%Y %m.%d. %H:%M:%S")
    to_save['TIME'] = _change_format(time, 'TIME')

    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(TABLE)
    
    for i in range(10):
        to_save['ID'] = to_save['TIME'] + ' ' + _random_string_generator(max(6,i*2))
        try:
            out = table.put_item(
                    Item=to_save,
                    ConditionExpression="attribute_not_exists(ID)"
                    )
            break
        except boto3.exceptions.botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException' and i!=9:
                continue
            raise

def read_all(TABLE, FILTER=tuple(), MAP=tuple()):
    """
    Reads all values from database (optionally filtered)
    FILTER must be tuple of callables returning Booleans (False means it's filtered out row)
    MAP list of map function 
    """
    assert isinstance(FILTER, tuple)
    assert all((hasattr(f,'__call__') for f in FILTER))

    def filter_dict(d):
        return all((func(d) for func in FILTER))
    def map_dict(d):
        return functools.reduce(lambda d, f: f(d), MAP, d)
    def change_dict(d):
        return {k:_change_format_back(v,k) for k,v in d.items() if k != 'ID'}

    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(TABLE)
    result = table.scan()['Items']

    return list(filter(filter_dict, map(map_dict, map(change_dict, result))))

"""
TODO
def to_csv(data, delimiter='\t', sort_key=lambda x: x['RESULT']):
    for values in sorted(data):
        delimiter.join()
"""

        




