from codetiming import Timer
import logging
from core.problem import Problem
from core.algorithm import greedy_alg
from core.utils import glob
from core.utils import pd

def phoenix_ga(
        stockpile,
        number_bench,
        number_cut,
        demand,
        greedy_alpha,
        direction,
        phoenix,
        ls,
        iteration,
        case,
        ):

    # input parameters
    stockpile = int(stockpile)
    number_bench = int(number_bench)
    number_cut = int(number_cut)
    demand = float(demand)
    greedy_alpha = int(greedy_alpha)
    direction = int(direction)
    phoenix = int(phoenix)
    ls = int(ls)
    iteration = int(iteration)
    case = int(case)

    # if greedy_alpha > 0:

    p = Problem(1, 1, stockpile, number_bench, number_cut, demand, 0, greedy_alpha,
                direction, phoenix, ls, case)
    p.clean()
    print('iteration {}: \n'.format(iteration))
    # t = Timer("example", text="Time spent: {:.2f}", logger=logging.warning)
    # t.start()
    A = greedy_alg(p, iteration)
    A.run()