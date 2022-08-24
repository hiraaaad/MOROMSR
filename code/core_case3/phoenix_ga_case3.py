from codetiming import Timer
import logging
from core_case3.problem import Problem
from core_case3.algorithm import greedy_alg_case3
from core_case3.utils import glob
from core_case3.utils import pd

def phoenix_ga_case3(
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
    A = greedy_alg_case3(p, iteration)
    A.run()