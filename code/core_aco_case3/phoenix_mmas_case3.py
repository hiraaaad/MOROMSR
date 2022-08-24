from core_aco_case3.problem import Problem_ACO
from core_aco_case3.algorithm import Algorithm_MMAS_Case3
from codetiming import Timer
import logging

def phoenix_mmas_case3(
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
        ):

    # input parameters
    stockpile = int(stockpile)
    number_bench = int(number_bench)
    number_cut = int(number_cut)
    demand = float(demand)
    direction = int(direction)
    phoenix = int(phoenix)
    ls = int(ls)
    iteration = int(iteration)
    alpha = float(alpha)
    beta = float(beta)
    rho = float(rho)
    num_ant = int(num_ant)
    num_gen = int(num_gen)
    case = int(case)

    # t = Timer("example", text="Time spent: {:.2f}", logger=logging.warning)
    p = Problem_ACO(1, 1, stockpile, number_bench, number_cut, demand, direction, phoenix, ls, case)
    # p.clean()
    MMAS_param = {'alpha': alpha, 'beta': beta, 'rho': rho, 'num_ant': num_ant, 'num_gen': num_gen}
    # t.start()
    A = Algorithm_MMAS_Case3(p, iteration, MMAS_param)
    A.run()
    # t.stop()
    # with open(A.log_str, "a") as logfile:
    #     logfile.write('runtime is {} \n'.format(t.last))




