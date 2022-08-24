import pandas as pd
import numpy as np
from core_aco.ant import Ant
import networkx as nx
from operator import attrgetter
import pathlib
import os
from codetiming import Timer
import logging


class Algorithm_MMAS():
    "colony includes multiple ants, it behaves as the colony iterator for the problem"

    def __init__(self, problem, iteration, MMAS_params):
        self.problem = problem
        self.problem.alpha = MMAS_params['alpha']
        self.problem.beta = MMAS_params['beta']
        self.problem.iteration = iteration
        self.rho = MMAS_params['rho']
        self.iteration = iteration
        self.number_nodes = len(self.problem.df_nodes)
        self.num_ant = MMAS_params['num_ant']
        self.colony = []
        self.termination_aco = False
        self.num_gen = MMAS_params['num_gen']
        self.ant_best_global = None
        self.setup_log()
        self.rng_seed = int(np.random.randint([1e6]))
        # self.rng_seed = int(304266)
        np.random.seed(self.rng_seed)
        print(self.rng_seed)
        self.count_local = 0
        self.evolution = []

        self.initialize_phermone()

        # with open(self.log_str, "a") as logfile:
        #     logfile.write('rng seed is {} \n'.format(self.rng_seed))
        #     print(self.rng_seed)
        #     logfile.write('iteration is {} \n'.format(iteration))

    def initialize_phermone(self):
        "initialize phermone values"
        self.df_ph = {k: 0.5 for k in self.problem.df_cost.keys()}

    def run(self):
        # t.start()
        gen = 1
        # upd_type = 0 # AS-Update
        while gen <= self.num_gen:
            self.colony = []
            count = 1
            # with open(self.log_str, "a") as logfile:
            #     logfile.write('Gen: {} | ants are constructing their solution \n'.format(gen))
            for _ in range(self.num_ant):
                ant = Ant(self.problem,'01-01-01-01-01',self.df_ph)
                if self.problem.case == 1:
                    ant.construct_solution_case_1()
                else:
                    ant.construct_solution_case_2()

                # t.start()
                #if self.problem.ls > 0:
                #     ant.local_search_iterative()
                #     with open(self.log_str, "a") as logfile:
                #         logfile.write('ant {} is doing a local search \n'.format(count))
                # t.stop()
                count += 1
                # print(count)
                self.colony.append(ant)
            # find best
            ant_best_iter = self.find_best()
            if self.problem.ls > 0:
                ant_best_iter.local_search_iterative()



            #     with open(self.log_str, "a") as logfile:
            #         logfile.write('best iteration ant is doing a local search \n'.format(count))
            self.update_pheromone(ant_best_iter)

            if self.ant_best_global is None:
                self.ant_best_global = ant_best_iter
            else:
                if ant_best_iter < self.ant_best_global:
                    self.ant_best_global = ant_best_iter

            # with open(self.log_str, "a") as logfile:
            #     logfile.write('gen: {}, best_local: {}, best_global: {} \n'.format(gen, ant_best_iter, self.ant_best_global))
                # logfile.write('tau max: {}, tau min: {} \n'.format(float(max(self.df_ph.values())), float(min(self.df_ph.values()))))
                # logfile.write('succesful local search: {} \n'.format(int(sum([x.count_local for x in self.colony]))))
            # print(gen, ant_best_iter, self.ant_best_global)
            gen+=1
            if np.mod(gen, 50) == 0:
                self.report_csv(0000)

            # print(np.random.get_state())

            self.count_local += int(sum([x.count_local for x in self.colony]))
            self.evolution.append((self.ant_best_global.solution.obj,self.ant_best_global.solution.viol))
            # print('Tau max is {}'.format(float(self.df_ph.max())))
        # print('designate best solution')
        # if self.problem.ls == 1:
        #     print(self.ant_best_global)
        #     self.ant_best_global.local_search_iterative()
        #     print(self.ant_best_global)
        # t.stop()
        # time_spent = t.last
        # self.report_csv(time_spent)
        # print(self.ant_best_global.solution.parcel_list)


        #     self.update_pheromone(upd_type)
        #     for ant in self.colony:
        #         print('Gen: {}, ant: {}',gen,ant)
        #

    def update_pheromone(self,ant_best_iter):
        "apply pheromone update"
        ant_best_iter.make_edges()
        self.df_ph = {key: max(round((1 - self.rho) * value,4), (1/self.number_nodes)) for key, value in self.df_ph.items()}
        for edge in ant_best_iter.edges:
            value = min(self.df_ph[edge] + self.rho, 1 - (1 / self.number_nodes))
            self.df_ph[edge]= round(value,4)
        # self.df_ph.Tau =
        # self.df_ph.Tau[self.df_ph.Tau<(1/self.number_nodes)]=

        # for edge in ant_best_iter.edges:
            # self.df_ph.loc[edge, 'Tau'] = round(,4)
        # for edge, tau in self.df_ph.iterrows():
        #     if edge in ant_best_iter.edges:
        #         self.df_ph.loc[edge, 'Tau'] = min((1-self.rho)*self.df_ph.loc[edge, 'Tau']+self.rho,1-(1/self.number_nodes))
        #     else:
        #         self.df_ph.loc[edge, 'Tau'] = max((1-self.rho)*self.df_ph.loc[edge, 'Tau'],(1/self.number_nodes))


    def find_best(self):
        cost = [x.solution.obj for x in self.colony]
        v = [x.solution.viol for x in self.colony]
        rank = np.lexsort((cost, v))
        return self.colony[rank[0]]

    def setup_log(self):
        if self.problem.case == 1:
            xx = '/result_case_1/'
            postfix = '_{}%'.format(self.problem.demand * 100)
        else:
            xx = '/result_case_2/'
            postfix = '_{}req'.format(self.problem.demand)

        x = 'MMAS_{}_{}_{}_{}{}'.format(self.problem.number_stockpile,
                                          self.problem.number_bench,
                                          self.problem.number_cut,
                                          self.problem.demand,
                                          postfix
                                          )
        if self.problem.ls > 0:
            if self.problem.ls == 1:
                x += '_local_swap'
            elif self.problem.ls == 2:
                x += '_local_ins'
            elif self.problem.ls == 3:
                x += '_local_inv'

        if self.problem.phoenix == True:
            pathlib.Path(os.getenv('PWD') + xx + x).mkdir(parents=True, exist_ok=True)
            self.log_directory = str(pathlib.Path(os.getenv('PWD') + xx)) + '/' + x
            log_str = '{}/{}_iter_{}'.format(self.log_directory, x, self.iteration)

        else:
            pathlib.Path(os.getenv("HOME") + xx + x).mkdir(parents=True, exist_ok=True)
            self.log_directory = str(pathlib.Path(os.getenv("HOME") + xx)) + '/' + x
            log_str = '{}/{}_iter_{}'.format(self.log_directory, x, self.iteration)

        self.log_str = log_str + '.log'
        self.log_csv = log_str + '.csv'
        self.log_csv_total = log_str + '_total.csv'
        self.log_csv_all = log_str + '_all.csv'
        self.log_csv_best = log_str + '_best.csv'
        self.log_evolution = log_str + '_evolution.csv'


    def report_csv(self,time_spent):
        table = pd.DataFrame(columns=['cut','direction','obj','Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2','tonnage'])
        for index, entry in enumerate(self.ant_best_global.solution.visited):
            cut = entry[0]
            a = [cut.name, entry[1], entry[2], cut.node_info['Al2O3'], cut.node_info['Fe'],
                 cut.node_info['Mn'], cut.node_info['P'], cut.node_info['S'], cut.node_info['SiO2'],
                 cut.node_info['Cut_Tonnage']]
            table.loc[index] = a
        table.set_index('cut')
        table.to_csv(self.log_csv)

        # ()
        #
        #
        #
        # table_visited.columns =
        # table_visited = table_visited.set_index('cut')
        # table_visited.to_csv(self.log_csv)

        df_total_csv = pd.DataFrame(columns=['cost', 'violation', 'reclaimed tonnage',
                                             'available tonnage','parcel_list','seed','time'])
        df_total_csv = df_total_csv.append(
            {'cost': float(self.ant_best_global.solution.obj),
             'violation': float(self.ant_best_global.solution.viol),
             'reclaimed tonnage': float(self.ant_best_global.solution.tonnage_so_far),
             'available tonnage': float(self.ant_best_global.total_capacity),
             'parcel_list': self.ant_best_global.solution.parcel_list,
             'seed':self.rng_seed,
             'time':time_spent,
             'count_local':self.count_local,
             },
            ignore_index=True)
        df_total_csv = df_total_csv.transpose()
        df_total_csv.to_csv(self.log_csv_total)



        df_evolution= pd.DataFrame(self.evolution, columns=['obj','viol'])
        df_evolution.to_csv(self.log_evolution)





