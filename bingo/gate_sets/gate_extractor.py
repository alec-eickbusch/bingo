import tensorflow as tf

#This function is a helper function for testing purposes.
#It takes a gate set, and given parameters for the gate,
#it extracts a single block.
def get_block_operator_from_gateset(gateset, parameter_values):
    #First, get the block operator
    opt_vars = {}
    for key, value in parameter_values.items():
        opt_vars[key] = tf.constant([[value]], dtype=tf.float32)
    op = gateset.batch_construct_block_operators(opt_vars)
    return tf.squeeze(op)