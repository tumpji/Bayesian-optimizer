#!/usr/bin/python3
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: gpyOptDomain.py
#  DESCRIPTION: This module creates classes representing optimization domains
#        USAGE: 1) Create global classes based on need, 
#               2) call GpyOptOption.convert_to_objects(*values) in order to optain (name, value) pairs
#               3) call GpyOptOption.convert_to_db_description(*values) in order to obtain (name: value) 
#                  dictionary ready to be saved into database
# REQUIREMENTS:
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
#      CREATED: 2019 04.05.
# =============================================================================

from abc import ABCMeta, abstractmethod
import operator
import numpy as np


class GpyOptOption(metaclass=ABCMeta):
    ''' abstract method, use only its classmethods '''
    closed = False
    instances = []
    raw_constraints = []
    constraints = []


    def __init__(self, name):
        assert not self.closed
        if name in [i.name for i in self.instances]:
            raise ValueError("Name shoud be unique \'{}\'".format(name))
        self.name = name
        GpyOptOption.instances.append(self)
        self.instances.sort(key=operator.attrgetter('name'))

    @classmethod
    def generate_domains(cls):
        """ Generate list of domains for GPyOpt """
        return [y._to_gpyopt_domain_dict() for y in cls.instances if y.single() is None]

    @classmethod
    def generate_constraints(cls):
        """ Generate list of constraints for GPyOpt """
        if len(cls.constraints):
            return cls.constraints
        return None

    @classmethod
    def convert_to_objects(cls, *args):
        """ converts inputs from GPyOpt to object (like func., floats, int...) """
        result = {}
        i_arg = 0
        for y in cls.instances:
            g = y.single()
            result[y.name] = y._index_to_object(args[i_arg]) if g is None else g[0]
            if g is None:
                i_arg += 1
        return result

    @classmethod
    def convert_to_db_description(cls, *args):
        """ converts inputs from GPyOpt to objects storable in database """
        result = {}
        i_arg = 0
        for y in cls.instances:
            g = y.single()
            result[y.name] = y._index_to_db_object(args[i_arg]) if g is None else g[1]
            if g is None:
                i_arg += 1
        return result

    @classmethod
    def tagit(cls, args):
        """ generate human readable list of a=value """
        result = []
        i_arg = 0
        for y in cls.instances:
            g = y.single()
            result.append( "{}={}".format(y.name, y._index_to_db_object(args[i_arg]) if g is None else g[1]) )
            if g is None:
                i_arg += 1
        return result

    @classmethod
    def convert_from_previous_results(cls, db, maximize=False):
        """ converts databse list of dicts to X,Y (initial inputs for optimization) """
        assert isinstance(db, list)
        assert all(isinstance(x, dict) for x in db)
        #synonyms = {'False': False, 'True': True}

        X = [] #2d numpy array (one per row)
        Y = []

        for row in db:
            act = np.zeros((1,len(cls.instances)), dtype=np.float32)
            error = False

            for i, instance in enumerate(cls.instances):
                if instance.name in row:
                    a = instance._from_db_to_index(row[instance.name])
                    if a is None:
                        error = True
                        break
                    act[0,i] = a
                elif instance.has_default():
                    act[0,i] = instance.default()
                else:
                    error = True
                    print("Coud not load row {}: \'{}\' have problem (try to make default index)".format(row, instance.name))
                    break
            if not error:
                X.append(act)
                if maximize:
                    Y.append(-row['RESULT'])
                else:
                    Y.append(row['RESULT'])

        if len(X) > 0:
            X = np.concatenate(X, axis=0)

            for dim, y in reversed(list(enumerate(cls.instances))):
                if y.single() is not None:
                    X = np.delete(X, dim, axis=1)

            Y = np.array(Y, dtype=np.float32)
            Y = np.expand_dims(Y, axis=1)
            return {'Y':Y, 'X':X}
        return {}

    @classmethod
    def finalize(cls):
        """ close posibility add new value + create constraints """
        closed = True

        # create constraints (indexes are now known and stable)
        index_dictionary = {}
        i = 0
        for ins in cls.instances:
            if ins.single() is None:
                index_dictionary[ins.name] = "x[:,{}]".format(i)
                i += 1

        for (c_name, c_def) in cls.raw_constraints:
            c = c_def.format( **index_dictionary )
            cls.constraints.append({'name': c_name, 'constraint': c})


    def has_default(): 
        """ is default value defined """
        return self.default is not None
    def default(self): 
        """ return default value (GpyOpt format) """
        return self.default
    @abstractmethod
    def single(self): 
        """ returns if is posible to ommit this value None - no, (v_gpyopt, v_db) - yes"""
        return None


    @classmethod
    def add_constraint(cls, name, definition):
        cls.raw_constraints.append((name, definition))

    @abstractmethod
    def _index_to_db_object(self, i): pass
    @abstractmethod
    def _index_to_object(self, i): pass
    @abstractmethod
    def _to_gpyopt_domain_dict(self): pass
    @abstractmethod
    def _from_db_to_index(self, dbobj): pass

    def force_values_to_others_if_equal_to(self, set_of_conditions, other_object, set_of_allowed_values):
        assert hasattr(set_of_conditions, '__iter__') and hasattr(set_of_allowed_values, '__iter__')
        assert not isinstance(other_object, Continuous) and isinstance(other_object, GpyOptOption)

        assert all(x in self.list_of_names for x in set_of_conditions) \
            if isinstance(self, CategoricalLabel) else \
            all(x in self.values for x in set_of_conditions)

        assert all(x in other_object.list_of_names for x in set_of_allowed_values) \
            if isinstance(other_object, CategoricalLabel) else \
            all(x in other_object.values for x in set_of_allowed_values)

        c = []
        for cval in set_of_conditions:
            c.append('{{{}}} == {}'.format(self.name, self._from_db_to_index(cval)))
        c = 'np.logical_or.reduce((' + ','.join(c) + ',))'


        d = []
        for dval in set_of_allowed_values:
            d.append('{{{}}} == {}'.format(other_object.name, other_object._from_db_to_index(dval)))
        d = 'np.logical_or.reduce((' + ','.join(d) + ',))'

        # implication

        condition = '-1*(np.logical_or(np.logical_not(' + c + '), ' + d + ')) + 0.001'
        self.add_constraint(None, condition)





class Continuous(GpyOptOption):
    ''' continuous variable '''
    def __init__(self, name, minimum, maximum, default=None):
        super(Continuous, self).__init__(name)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        assert default is None or isinstance(default, float)
        self.default = default

    def _index_to_db_object(self, i): return i
    def _index_to_object(self, i): return i
    def _to_gpyopt_domain_dict(self):
        return {'name': self.name, 'type': 'continuous', 'domain': (self.minimum, self.maximum)}
    def _from_db_to_index(self, db):
        if self.minimum <= db <= self.maximum: return db
        return None
    def single(self): return None

class Discrete(GpyOptOption):
    def __init__(self, name, values, default=None):
        assert hasattr(values, '__iter__')
        super(Discrete, self).__init__(name)
        self.values = list(values)
        assert default is None or default in self.values
        self.default = default

    def _index_to_db_object(self, i): return i
    def _index_to_object(self, i): return i
    def _to_gpyopt_domain_dict(self):
        return {'name': self.name, 'type': 'discrete', 'domain': self.values}
    def _from_db_to_index(self, db):
        if db in self.values:
            return db
        return None

    def single(self): return (self.values[0], self.values[0]) if len(self.values) <= 1 else None

    

class DiscreteInt(Discrete):
    ''' discrete variable e.g. 1,2,3,4 but not 1.5, 
        values are somewhat semanticaly sorted by its numeracal values 
        e.g. 
            svm polynomial kernel: p=2,3,4,5  overall there shoud be decline after reaching optimum (overtraining)
    '''
    def __init__(self, name, values, default=None):
        assert all(isinstance(v, int) for v in values)
        super(DiscreteInt, self).__init__(name, values, default=default)

    def _index_to_db_object(self, i): return int(i)
    def _index_to_object(self, i): return int(i)

    def single(self): return (self.values[0], self.values[0]) if len(self.values) <= 1 else None




class Categorical(GpyOptOption):
    ''' categorical value, values are not sorted by its order or values
        they can be integers, strings, booleans or real numbers
        e.g. 
            'Adam', 'SGD', 'RMS-Prop'
            1,2,3,4 (in meaning of 4-th algorithm or picture ...)
    '''
    def __init__(self, name, values, default=None):
        super(Categorical, self).__init__(name)
        assert isinstance(values, (list, tuple))
        assert all((isinstance(x, (str, int, float, bool)) for x in values))
        self.values = list(values)
        assert default is None or default in self.values
        self.default = default


    def _index_to_db_object(self, i):
        return self.values[int(i)]
    def _index_to_object(self, i):
        return self.values[int(i)]
    def _to_gpyopt_domain_dict(self):
        return {'name': self.name, 'type': 'categorical', 'domain': list(range(len(self.values)))}
    def _from_db_to_index(self, db):
        if db in self.values:
            return self.values.index(db)
        return None

    def single(self): return (self.values[0], self.values[0]) if len(self.values) <= 1 else None

class CategoricalLabel(GpyOptOption):
    '''
        Categorical with label as an id, you need to provide function
        e.g.
            ('MinMaxNorm', lambda: MinMaxScaler(-0.5,0.5)), ...
    '''
    def __init__(self, name, name_object_pairs, default_index=None):
        """ ('Name of function', function_callable) """
        assert isinstance(name_object_pairs, (list, tuple))
        assert all( (isinstance(x, tuple) for x in name_object_pairs) )
        assert all( (len(x) == 2 for x in name_object_pairs) )
        assert all( (isinstance(x[0], str) for x in name_object_pairs) )
        super(CategoricalLabel, self).__init__(name)

        self.list_of_names = [ x[0] for x in name_object_pairs ]
        self.list_of_objects = [ x[1] for x in name_object_pairs ]

        assert default_index is None or default_index < len(self.list_of_names)
        self.default = float(default_index) if default_index is not None else None

    def _index_to_db_object(self, i):
        return self.list_of_names[int(i)]
    def _index_to_object(self, i):
        return self.list_of_objects[int(i)]
    def _to_gpyopt_domain_dict(self):
        return {'name': self.name, 'type': 'categorical', 'domain': list(range(len(self.list_of_names)))}
    def _from_db_to_index(self, db):
        if db in self.list_of_names:
            return self.list_of_names.index(db)
        return None

    def single(self): return (self.list_of_objects[0], self.list_of_names[0]) if len(self.list_of_objects) <= 1 else None
    

class Layers:
    """
        behaviour is one of ['decrese_or_equal', 'decrese', 'no space']
            decrese_or_equal: 500-500 ok
            decrese: 500-499 ok
            no space 200-1-500-3 ok

        optionaly you can generate variables for coresponding layers with:
            generate_for_each_layer_bool, generate_for_each_layer_float, generate_for_each_layer_categorical 

        maximum_neurons (including value)
    """
    def __init__(self, name, maximum_layers, maximum_first_layer, 
            maximum_neurons=None, behaviour='decrese_or_equal', allow_empty_first_layer=False):
        assert isinstance(maximum_layers, int) and maximum_layers > 0
        assert isinstance(maximum_first_layer, int) and maximum_first_layer > 0
        assert maximum_neurons is None or isinstance(maximum_neurons, int)

        self.name = name
        self.maximum_layers = maximum_layers
        self.extra_vars = 1

        for lno in range(maximum_layers):
            lname = name + "_" + str(lno)
            if lno == 0 and not allow_empty_first_layer:
                DiscreteInt(lname, range(1, maximum_first_layer))
                # houd be bigger than 0 => ok
            else:
                DiscreteInt(lname, range(0, maximum_first_layer))
                # shoud be smaller than previous
                previous_lname = name + "_" + str(lno-1)
                if behaviour == 'decrese_or_equal':
                    c = "{{{}}} - {{{}}} - 0.001".format(lname, previous_lname)
                elif behaviour == 'decrese':
                    c = "np.logical_not(np.logical_or( ({{{}}} - {{{}}}) < 0, np.logical_and({{{}}} == 0, {{{}}} == 0)))*1 - 0.001".format(
                            lname, previous_lname, lname, previous_lname )
                elif behaviour == 'no space':
                    c = "np.logical_and(( {{{}}} > 0 ), ( {{{}}} == 0 ))*1 - 0.001".format(lname, previous_lname)
                else:
                    raise NotImplementedError
                GpyOptOption.add_constraint("{}_const".format(lname), c)

        if maximum_neurons is not None:
            c = "+".join( "{{{}}}".format(name + "_" + str(lno)) for lno in range(maximum_layers) ) 
            GpyOptOption.add_constraint(name + "_const_numneu", c + "-{} - 0.001".format(maximum_neurons))

    def _get_layer_names(self):
        """ returns generator that generates all variable names associated with layers """
        return [self.name + '_' + str(lno) for lno in range(self.maximum_layers)]

    def get_layers_from_object_dict(self, odict, reduce_minimum=True):
        t = [ odict[n] for n in self._get_layer_names() ]
        if reduce_minimum:
            t = list(filter(lambda s: s > 0, t))
        return t

    def _generate_for_each_layer(self, group_name, add, function):
        names = []
        for lno in range(self.maximum_layers + add):
            name = "{}_{}_{}".format(self.name, lno, group_name)
            function(name)
            names.append(name)
        self.extra_vars += 1
        return names


    def generate_for_each_layer_bool(self, group_name, add=0, default=None):
        """ creates boolean variables (for each layer) and returns it's names """
        return self._generate_for_each_layer(group_name, add, lambda name: Categorical(name, [False, True], default=default))

    def generate_for_each_layer_float(self, group_name, minimum, maximum, add=0, default=None):
        """ creates continuous variables (for each layer) and returns it's names """
        return self._generate_for_each_layer(group_name, add, lambda name: Continuous(name, minimum, maximum, default=default))

    def generate_for_each_layer_categorical(self, group_name, values, add=0, default=None):
        """ creates categorical variable (for each layer) and returns it's names """
        return self._generate_for_each_layer(group_name, add, lambda name: Categorical(name, values, default=default))




    


