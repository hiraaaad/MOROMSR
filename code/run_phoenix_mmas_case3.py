import sys
[
    stockpile,
    number_bench,
    number_cut,
    demand,
    direction,
    phoenix,
    ls,
    iteration,
    case,
    alpha,
    beta,
    rho,
    num_ant,
    num_gen,
]=sys.argv[1:]
from core_aco_case3.phoenix_mmas_case3 import phoenix_mmas_case3
phoenix_mmas_case3(stockpile, number_bench, number_cut, demand, direction, phoenix, ls, iteration, case, alpha, beta, rho, num_ant, num_gen)