import numpy as np
from test_gradient_for_python_layer import test_gradient_for_python_layer

# set the inputs
input_names_and_values = [('in_cont', np.random.randn(64, 5))]
output_names = ['out1']
py_module = 'mvn'
py_layer = 'MVNLayer'
param_str = ''
propagate_down = [True]

# call the test
test_gradient_for_python_layer(input_names_and_values, output_names, py_module, py_layer, param_str, propagate_down)

# you are done!