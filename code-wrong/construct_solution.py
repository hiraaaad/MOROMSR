# this script constructs a valid solution for an agent (ant)
# 1. initiate from the entry point
# 2. look at the neighborhood and choose the successor cut
# 3. repeat it until the termination criterion
## Greedy algorithm
import pickle
from utils import np, nx
from algorithms import GA
from psrh_algorithm import PSRH
from mmas import MMAS
from problem import Problem

problem = Problem(scenario = 2, number_demand = 2, number_stockpile = 4, greedy_factor = 2, parcel = True, local = True)

# psrh = PSRH(problem=problem)
# psrh.evolve()
#
ga = GA(problem=problem)
ga.evolve()

# mmas = MMAS(problem=problem)
# mmas.evolve()
