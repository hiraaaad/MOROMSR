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
from core_aco.phoenix_mmas import phoenix_mmas
phoenix_mmas(stockpile, number_bench, number_cut, demand, direction, phoenix, ls, iteration, case, alpha, beta, rho, num_ant, num_gen)