import os

import sympy
from gplearn.genetic import SymbolicRegressor
import matplotlib.pyplot as plt
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
f_a = trees['f_a'][mask].reshape((-1, 1))
f_c = trees['f_c'][mask].reshape((-1, 1))
X = np.concatenate([z, dm_mass], axis=1)
names = ['z', 'M_h']
y = np.concatenate([f_a, f_c], axis=1)
y = f_c

fig, ax = plt.subplots(1, dpi=200)
ax.plot(X[:, 0], f_c, '.')
# ax.plot(X[:, 1], f_c, '.')
# ax.set_xscale('log')

ax.set_yscale('log')
plt.show()
plt.close()


# TODO: Smooth? dataset
mask = np.random.randint(X.shape[0], size=8000)
X = X[mask]
y = y[mask]

est_gp = SymbolicRegressor(population_size=5000,
                           generations=20,
                           p_crossover=0.7,
                           p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05,
                           p_point_mutation=0.1,
                           max_samples=0.9,
                           verbose=1,
                           parsimony_coefficient=0.0005,
                           random_state=0,
                           feature_names=names)
est_gp.fit(X, y)
converter = {
    'sub': lambda x, y : x - y,
    'div': lambda x, y : x/y,
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x + y,
    'neg': lambda x    : -x,
    'pow': lambda x, y : x**y
}
best_programs = sorted(est_gp._programs[-1], key=lambda p: p.fitness_)
exps = set()
print('Fitness        Loss           Len       Function')
i = 0
while len(exps) < 3:
    p = best_programs[i]
    exp = sympy.sympify(str(p), locals=converter)
    exp = sympy.simplify(exp)
    i += 1
    if exp in exps:
        continue
    exps.add(exp)
    print(str(round(p.fitness_, 5)).ljust(15), end='')
    print(str(round(p.raw_fitness_, 5)).ljust(15), end='')
    print(str(len(p.program)).ljust(10), end='')
    print(exp)

exit()


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
