import os

import numpy as np
from pysr import PySRRegressor

import helpers
from helpers import log

config = helpers.Config()
log(f'Loading data')
data_dir = config.get_generated_data_dir() + 'trees/'
trees = {filename.replace('.npy', ''): np.load(data_dir+filename) for filename in os.listdir(data_dir)}

log(f'Creating input and output arrays')
mask = (trees['dm_mass'] != 0) & (trees['f_a'] != -1)
z = trees['z'][mask].reshape((-1, 1))
dm_mass = trees['dm_mass'][mask].reshape((-1, 1))
f_a = trees['f_a'][mask]
X = np.concatenate([z, dm_mass], axis=1)
names = ['z', 'M_h']
y = f_a

# TODO: Smooth? dataset
mask = np.random.randint(X.shape[0], size=8000)
X = X[mask]
y = y[mask]

log(f'Initialising regressor')
model = PySRRegressor(
    procs=4,
    model_selection="best",  # Result is mix of simplicity+accuracy
    niterations=100,
    binary_operators=["plus", "sub", "mult", "pow", "div"],
    unary_operators=[
        "log10_abs",
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
	# ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
    # TODO: Denoise significantly increases runtime
    # denoise=True,
)

log(f'Training regressor')
model.fit(X, y, variable_names=names)

print(model)

log(f'Script finished')
