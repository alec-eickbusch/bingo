TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
END_OPT_STRING = "\n" + "=" * 60 + "\n"
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # suppress warnings

import bingo.optimizer.tf_quantum as tfq
import qutip as qt
import datetime
import time
from typing import List, Dict


class GateSet:

    """
    This class is intended to be a barebones implementation of a specific gate set. Here, we only want to define the blocks in the gate
    set and the parameters that will be optimized. This class will be passed to the GateSynthesizer class which will call the optimizer
    your choice to optimize the GateSet parameters.
    """

    def __init__(
        self, name="GateSet", **kwargs
    ):  # some of the above may not be necessary. i.e. dimension, N_blocks, n_parameters are implicit in some of the defs below. think about this
        self.parameters = {
            "name": name,
        }
        self.parameters.update(kwargs)

    @property
    def parameter_names(self):
        """
        Returns name of parameters for this gate set in order. 
        
        Returns
        -----------
        list of strings
        """
        pass

    def randomization_ranges(self):
        """
        For each parameter, specify the range to use for random initialization.
        
        Returns
        -----------
        dictionary of tuples.
        Keys: parameter names
        items: tuple of (low, high) range for randomization.
        """
        pass

    def modify_parameters(self, **kwargs):
        # currently, does not support changing optimization type.
        # todo: update for multi-state optimization and unitary optimziation
        parameters = kwargs
        for param, value in self.parameters.items():
            if param not in parameters:
                parameters[param] = value
        self.__init__(**parameters)

    @tf.function
    def batch_construct_block_operators(
        self, opt_params: Dict[str, tf.Variable], *args
    ):
        """
        This function must take a dict of tf.Variable defined in the same order as randomize_and_set_vars()
        and construct a batch of block operators. Note that the performance of the optimization depends heavily
        on your implementation of this function. For the best performance, do everything with vectorized operations
        and decorate your implementation with @tf.function.

        Parameters
        -----------
        opt_params  :   dict of optimization parameters. This dict must be of the same length
                        as the one defined in ``randomize_and_set_vars``. Each element in the dict
                        should be of dimension (N_blocks, N_multistart).
        
        Returns
        -----------
        tf.tensor of block operators U of size (N_multistart, U.shape)
        """

        pass

    def preprocess_params_before_saving(
        self, opt_params: Dict[str, tf.Variable], *args
    ):
        """
        When defined, this function defines a way to process the optimization parameters before they are saved
        in the h5 file. See the ECD_gate_set for an example of this in action.

        Parameters
        -----------
        opt_params  :   Dict of optimization parameters. This dict must be of the same length
                        as the one defined in ``randomize_and_set_vars``. Each element in the dict
                        should be of dimension (N_blocks, N_multistart).

        Returns
        -----------
        Dict of tf.Variable. This dict does not need to be the same length as opt_params. Conversion to numpy 
        arrays is handled in the batch optimizer.
        
        """

        return opt_params
