from core.utils import np, os, pd, Utils, islice
from core.problem import Problem
from core.node import Solution
from codetiming import Timer
import logging
import copy
import pathlib
import glob

class greedy_alg:

    def __init__(self, problem, iter):
        self.utils = Utils(problem)
        self.utils.problem.greedy_alpha_iter = iter
        # print(iter)

        if len(self.utils.problem.directions)>1:
            direction = 'Bi'
        else:
            if self.utils.problem.directions[0]=='SN':
                direction = 'SN'
            else:
                direction = 'NS'

        self.solution = Solution(None)
        self.reclaimed_cuts = {}
        self.reclaimed_cuts_keys = []
        self.df_nodes = problem.df_nodes
        self.setup_accessible_nodes()
        self.setup_log()
        self.count_local = 0


        self.move_count = 1
        self.pos_reclaimer = None
        self.termination = False
        self.total_capacity = np.round(self.utils.problem.demand * 1e5)
        # print(self.total_capacity)
        # self.constraint_violation = 0

        # fill log
        with open(self.log_str, "a") as logfile:
            logfile.write('rng seed is {} \n'.format(self.utils.problem.rng_seed))
            if self.utils.problem.greedy_alpha>0:
                logfile.write('iteration is {} \n'.format(iter))

    def reclaim_initial(self,initial_node):
        node = self.utils.problem.df_nodes_dic[initial_node]
        self.solution.obj = node.cost_reclaim / node.cut_tonnage
        self.solution.tonnage_so_far = node.cut_tonnage
        direction = 'SN'
        # if initial_node in self.accessible_nodes[direction]:
        del self.accessible_nodes[direction][initial_node]
        node_info = node.node_info
        # node_info['Direction']=direction
        for key in self.solution.visited_info:
            self.solution.visited_info[key].append(node_info[key])
        # self.solution.visited_info = self.solution.visited_info.append(node_info)
        append_node = (node,direction, self.solution.obj)
        self.solution.visited.append(append_node)
        self.reclaimed_cuts[node.name]=append_node
        self.reclaimed_cuts_keys.append(node.name)
        self.solution.make_parcel(node)

    # def report_csv(self, cut, direction, cost, violation, tonnage):
    #     self.df_csv = self.df_csv.append({'Node':cut, 'Direction':direction, 'Cost':cost, 'Constraint violation':violation, 'Tonnage reclaimed': tonnage}, ignore_index=True)

    def setup_log(self):
        if self.utils.problem.greedy_alpha ==0:
            prefix = 'GA'
        else:
            prefix = 'RGA'

        if self.utils.problem.case == 1:
            xx = '/result_case_1/'
            postfix = '_{}%'.format(self.utils.problem.demand*100)
        else:
            xx = '/result_case_2/'
            postfix = '_{}req'.format(self.utils.problem.demand)

        x = '{}_{}_{}_{}_{}{}'.format(prefix,self.utils.problem.number_stockpile,
                                       self.utils.problem.number_bench,
                                       self.utils.problem.number_cut,
                                       self.utils.problem.demand,
                                       postfix,
                                       )

        if self.utils.problem.greedy_alpha > 0:
            x += '_alpha_{}'.format(int(self.utils.problem.greedy_alpha))

        if self.utils.problem.ls > 0:
            if self.utils.problem.ls == 1:
                x += '_local_swap'
            elif self.utils.problem.ls == 2:
                x += '_local_ins'
            elif self.utils.problem.ls == 3:
                x += '_local_inv'


        if self.utils.problem.phoenix:
            pathlib.Path(os.getenv('PWD') + xx + x).mkdir(parents=True, exist_ok=True)
            self.log_directory = str(pathlib.Path(os.getenv('PWD') + xx)) + '/' + x
        else:
            pathlib.Path(os.getenv("HOME") + xx + x).mkdir(parents=True, exist_ok=True)
            self.log_directory = str(pathlib.Path(os.getenv("HOME") + xx)) + '/' + x

        if self.utils.problem.greedy_alpha == 0:
            log_str = '{}/{}'.format(self.log_directory, x)
        else:
            log_str = '{}/{}_iter_{}'.format(self.log_directory, x, self.utils.problem.greedy_alpha_iter)

        self.log_str = log_str + '.log'
        self.log_csv = log_str + '.csv'
        self.log_csv_total = log_str + '_total.csv'
        self.log_csv_all = log_str + '_all.csv'
        self.log_csv_best = log_str + '_best.csv'

    def run(self):
        t = Timer("example", text="Time spent: {:.2f}")
        t.start()
        if self.utils.problem.case == 1:
            self.run_case_1()
        else:
            self.run_case_2()

        if self.utils.problem.ls > 0:
            self.local_search_iterative()
        t.stop()
        time_spent = t.last
        self.report(time_spent)


    def run_case_1(self):
        t = Timer("example", text="Time spent: {:.2f}", logger=logging.warning)
        self.total_capacity = np.round(self.utils.problem.demand*self.utils.problem.total_tonnage,4)
        self.reclaim_initial('01-01-01-01-01')
        self.termination = False
        while self.termination == False:
            neighbor_lst = []
            for direction in self.utils.problem.directions:
                neighbor_lst.extend(self.find_neighborhood(direction))
            for direction in self.utils.problem.directions:
                neighbor_lst.extend(self.accessible_nodes[direction].values())
            fitness_candidates = self.evaluate_node(neighbor_lst)
            next_idx = self.utils.greedy_selection_constrained(fitness_candidates) #,cost_candidate)
            next_node = neighbor_lst[next_idx][0]
            next_node_direction = neighbor_lst[next_idx][1]
            self.solution.obj += fitness_candidates[next_idx][0] #cost
            # self.solution.viol = fitness_candidates[next_idx][1] #violation
            self.solution.parcel_list[-1].penalty_mineral_avg = fitness_candidates[next_idx][2] #average_limits
            self.solution.parcel_list[-1].length += 1
            self.solution.parcel_list[-1].penalty = fitness_candidates[next_idx][1] #violation
            self.solution.viol = sum([x.penalty for x in self.solution.parcel_list])
            node_info = next_node.node_info
            self.solution.tonnage_so_far += next_node.cut_tonnage
            for key in self.solution.visited_info:
                self.solution.visited_info[key].append(node_info[key])
            # self.solution.visited_info = self.solution.visited_info.append(node_info)
            append_node = (next_node, next_node_direction, fitness_candidates[next_idx][0])
            self.solution.visited.append(append_node)
            self.reclaimed_cuts[next_node.name] = append_node
            self.reclaimed_cuts_keys.append(next_node.name)
            self.move_count += 1
            del(neighbor_lst[next_idx])
            self.update_accessible_nodes(neighbor_lst, next_node.name) # accessible nodes in the stockyard for next iteration
            if np.round(self.solution.tonnage_so_far,4) >= np.round(self.total_capacity,4):
                self.termination = True
                # self.solution.parcel_list[-1].penalty = self.solution.viol
                self.solution.parcel_list[-1].end = len(self.solution)-1
                L = [x[0] for x in self.solution.visited]
                self.solution.parcel_list[-1].visited = L[self.solution.parcel_list[-1].start:self.solution.parcel_list[-1].end+1]

    def run_case_2(self):
        # t = Timer("example", text="Time spent: {:.2f}", logger=logging.warning)
        # self.total_capacity = np.round(self.utils.problem.percentage*self.utils.problem.total_tonnage,4)
        self.reclaim_initial('01-01-01-01-01')
        for demand in range(self.utils.problem.demand):
            self.termination = False
            while self.termination == False:
                neighbor_lst = []
                for direction in self.utils.problem.directions:
                    neighbor_lst.extend(self.find_neighborhood(direction))
                for direction in self.utils.problem.directions:
                    neighbor_lst.extend(self.accessible_nodes[direction].values())
                fitness_candidates = self.evaluate_node(neighbor_lst)
                next_idx = self.utils.greedy_selection_constrained(fitness_candidates) #,cost_candidate)
                next_node = neighbor_lst[next_idx][0]
                next_node_direction = neighbor_lst[next_idx][1]
                self.solution.obj += fitness_candidates[next_idx][0] #cost
                # self.solution.viol = fitness_candidates[next_idx][1] #violation
                self.solution.parcel_list[-1].penalty_mineral_avg = fitness_candidates[next_idx][2] #average_limits
                self.solution.parcel_list[-1].length += 1
                self.solution.parcel_list[-1].penalty = fitness_candidates[next_idx][1] #violation
                self.solution.viol = sum([x.penalty for x in self.solution.parcel_list])
                node_info = next_node.node_info
                self.solution.tonnage_so_far += next_node.cut_tonnage
                for key in self.solution.visited_info:
                    self.solution.visited_info[key].append(node_info[key])
                # self.solution.visited_info = self.solution.visited_info.append(node_info)
                append_node = (next_node, next_node_direction, fitness_candidates[next_idx][0])
                self.solution.visited.append(append_node)
                self.reclaimed_cuts[next_node.name] = append_node
                self.reclaimed_cuts_keys.append(next_node.name)
                self.move_count += 1
                del(neighbor_lst[next_idx])
                self.update_accessible_nodes(neighbor_lst, next_node.name) # accessible nodes in the stockyard for next iteration
                # if np.round(self.solution.tonnage_so_far,4) >= np.round(self.total_capacity,4):
                if np.round(self.solution.tonnage_so_far,4) >= 1e5*(demand+1):
                    self.termination = True
                    # self.solution.parcel_list[-1].penalty = self.solution.viol
                    self.solution.parcel_list[-1].end = len(self.solution)-1
                    L = [x[0] for x in self.solution.visited]
                    self.solution.parcel_list[-1].visited = L[self.solution.parcel_list[-1].start:self.solution.parcel_list[-1].end+1]
                    # print(len(self.solution))
                    if demand+2 <= self.utils.problem.demand:
                        self.solution.make_parcel(None)

    def report(self,time_spent):
        # with open(self.utils.problem.log_str, "a") as logfile:
        #     # logfile.write('Total penalized cost is {} \n'.format(float(self.cost_so_far+self.utils.problem.penalty_coeff*self.constraint_violation)))
        #     logfile.write('Total cost is {} \n'.format(float(self.cost_so_far)))
        #     logfile.write('Total tonnage is {} \n'.format(float(self.tonnage_so_far)))
        #     logfile.write('Total violation is {} \n'.format(float(self.constraint_violation)))
        #     logfile.write('Total capacity is {} \n'.format(self.total_capacity))

        df_total_csv = pd.DataFrame(columns=['cost',
                                             'violation',
                                             'reclaimed tonnage',
                                             'available tonnage',
                                             'parcel_list',
                                             'seed',
                                             'time',
                                             'count_local',])
        df_total_csv = df_total_csv.append(
            {'cost': float(self.solution.obj), 'violation': float(self.solution.viol),
             'reclaimed tonnage': float(self.solution.tonnage_so_far),
             'available tonnage': float(self.total_capacity),
             'parcel_list':self.solution.parcel_list, 'seed':self.utils.problem.rng_seed,
             'time':time_spent, 'count_local':self.count_local}, ignore_index=True,)

        df_total_csv = df_total_csv.transpose()

        # table_visited = pd.DataFrame(self.solution.visited)
        # table_visited.columns = ['Node', 'Direction', 'Cost']
        # table_visited = table_visited.set_index('Node')
        # table_visited.to_csv(self.log_csv)

        table = pd.DataFrame(columns=['cut','direction','obj','Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2','tonnage'])
        for index, entry in enumerate(self.solution.visited):
            cut = entry[0]
            a = [cut.name, entry[1], entry[2], cut.node_info['Al2O3'], cut.node_info['Fe'],
         cut.node_info['Mn'], cut.node_info['P'], cut.node_info['S'], cut.node_info['SiO2'],cut.node_info['Cut_Tonnage']]
            table.loc[index] = a
        table.set_index('cut')
        table.to_csv(self.log_csv)






        # table.append()

        # self.df_csv.to_csv(self.log_csv, index=False)
        df_total_csv.to_csv(self.log_csv_total)

        with open(self.log_str, "a") as logfile:
            logfile.write('succesful local search: {} \n'.format(int(self.count_local)))

    def setup_accessible_nodes(self):
        self.accessible_nodes = {k: {} for k in self.utils.problem.directions}
        for k,v in self.utils.problem.stockpile_entry.items():
            for node_name in v:
                value = (self.utils.problem.df_nodes_dic[node_name], k)
                self.accessible_nodes[k].update({node_name:value})

    def update_accessible_nodes(self, neighbors, next_node_name):
        for direction in self.utils.problem.directions:
            if next_node_name in self.accessible_nodes[direction]:
                del self.accessible_nodes[direction][next_node_name]

        for node in neighbors:
            key = node[0].name
            direction = node[1]
            if key not in self.accessible_nodes[direction]:
                self.accessible_nodes[direction].update({key: node})
            if key in self.accessible_nodes[direction] and key == next_node_name:
                del self.accessible_nodes[direction][next_node_name]
        # for node in neighbors:
        #     if node[0] not in self.accessible_nodes.index:
        #         self.accessible_nodes.loc[node[0]] = node[1]
        #     else:
        #         if node[1] != self.accessible_nodes.loc[node[0]]['Direction']:
        #             self.accessible_nodes.loc[node[0]] = node[1]

    # # def local_search_iterative_demand(self,demand):
    # #     initial_solution = demand
    # #     termination = False
    # #     while termination is False:
    # #         local_solution = self.utils.local_search(initial_solution.visited)
    # #         local_solution.viol = initial_solution.viol
    # #         if local_solution < initial_solution:
    # #             self.count_local += 1
    # #             print(local_solution,initial_solution)
    # #             initial_solution = local_solution
    # #         else:
    # #             termination = True
    # #     return initial_solution
    #
    def local_search_iterative(self):
        # t = Timer("example", text="Time spent: {:.2f}", logger=logging.warning)
        # t.start()
        self.solution.reclaimed_cuts_keys = self.reclaimed_cuts_keys
        initial_solution = self.solution
        termination = False
        # tonnage_so_far = initial_solution.t
        while termination is False :
            local_solution = self.utils.local_search(initial_solution)
            if local_solution < initial_solution:
                self.count_local += 1
                # print(local_solution,initial_solution)
                initial_solution = local_solution
            else:
                termination = True
        self.solution = initial_solution
        # t.stop()


    def find_neighborhood(self, direction):
        # find neighbors in the current stockpile & add the accessible nodes to the end
        lst = []
        (pos_reclaimer, direction_reclaimer) = self.pos_current
        neighbor_list = self.utils.problem.df_nodes_dic[pos_reclaimer].prec_1[direction_reclaimer]
        if len(neighbor_list) > 0:
            for node in list(neighbor_list.Job_Step_2):
                if not self.reclaimed(node) and node not in self.accessible_nodes[direction].keys():
                    prec = self.utils.problem.df_nodes_dic[node].prec[direction]
                    if len(prec) > 0:
                        if all([self.reclaimed(x) for x in list(prec.Job_Step_1)]):
                            lst.append(node)
        output = []
        for node in lst:
            # output.append((node, direction))
            output.append((self.utils.problem.df_nodes_dic[node], direction))
        return output

    def evaluate_node(self, neighbors):
        fitness_candidates = []

        for neighbor in neighbors:
            t = Timer("example", text="Time spent: {:.2f}", logger=logging.warning)
            neighbor_name = neighbor[0].name
            neighbor_direction = neighbor[1]
            # t.start()
            neighbor_info = neighbor[0].node_info
            penalty_neighbor, average_limits = self.calc_penalty(neighbor_info)
            (pos_reclaimer, direction_reclaimer) = self.pos_current
            cost_move = self.utils.problem.df_cost[(pos_reclaimer, neighbor_name,
                                              direction_reclaimer, neighbor_direction)]['Cost']
            cost_reclaim = neighbor_info.Cost
            cost_total = cost_move + cost_reclaim
            cost_neighbor = cost_total / neighbor_info.Cut_Tonnage  # + self.problem.penalty_coeff*penalty_neighbor
            cost_neighbor = round(cost_neighbor, 6)
            penalty_neighbor = round(penalty_neighbor, 6)
            # penalty_report = round(penalty_report, 6)
            fitness_candidates.append((float(cost_neighbor), penalty_neighbor, average_limits))  # , penalty_report))
            # t.stop()
        return fitness_candidates

    def calc_penalty(self, neighbor_info):
        penalty_window = 0
        neighbor_info_properties = np.array([neighbor_info[x] for x in ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']])
        # add_penalty:
        n = self.solution.parcel_list[-1].length
        if n > 0:
            avg = self.solution.parcel_list[-1].penalty_mineral_avg
            average_limits = avg + ((neighbor_info_properties - avg)/ (n+1))
        else:
            average_limits = neighbor_info_properties
            self.solution.parcel_list[-1].start = len(self.solution)
            # if len(self.solution) == 1:
            #     self.solution.parcel_list[-1].start = 1
            # else:
            #     +1
        (penalty_lower, penalty_upper) = self.utils.calc_penalty_upper_lower(average_limits)
        penalty_neighbor = penalty_lower + penalty_upper + penalty_window
        # if len(self.solution)==88:
        #     print('CHECK')
        return (penalty_neighbor, average_limits)


        # penalty_mineral = 0
        #
        # self.utils.problem.lower_limits_array
        # self.utils.problem.lower_upper_array
        # penalty_window = 0
        # visited_temp = {k: [] for k in self.utils.problem.limits_lower}
        # # start = self.demands[-1][0]
        # # if start == 1:
        # #     start = 0
        # # len_demand = len(self.solution) - start
        # for k in visited_temp:
        #         # visited_temp[k].extend(self.solution.visited_info[k][start:])
        #         visited_temp[k].extend(self.solution.visited_info[k])
        #         visited_temp[k].append(neighbor_info[k])
        #
        # # if len(self.solution) > 3:
        # # # if len_demand > 3:
        # #     visited_temp_window = {k: v[-3:] for k, v in visited_temp.items()}
        # #     visited_temp_array = np.array([np.mean(v) for k, v in visited_temp_window.items()],dtype=float)
        # #     violation_window_lower = np.divide(visited_temp_array, self.utils.problem.lower_limits_window_array,
        # #                             out=np.ones(visited_temp_array.shape, dtype=float)*np.finfo(np.float32).max,
        # #                             where=self.utils.problem.lower_limits_window_array !=0) - 1
        # #     penalty_lower_window = abs(sum([x if x < 0 else 0 for x in violation_window_lower]))
        # #     violation_window_upper = (visited_temp_array / self.utils.problem.upper_limits_window_array) - 1
        # #     penalty_upper_window = sum([x if x > 0 else 0 for x in violation_window_upper])
        # #     penalty_window = penalty_lower_window + penalty_upper_window




    @property
    def pos_current(self):
        # return (self.solution.visited[-1][0].name, self.solution.visited[-1][1])
        index = self.reclaimed_cuts_keys[-1]
        return (index, self.reclaimed_cuts[index][1])

            # return (self.demands[-1].visited[-1][0].name,
        # return (self.solution.visited_info.iloc[-1].name, self.solution.visited_info.iloc[-1].Direction)

    def reclaimed(self, node):
        return node in self.reclaimed_cuts_keys
        # return node in [x[0].name for x in self.solution.visited]
        # return node in list(self.solution.visited_info.index)

    def solution_checker(self, csv_file, problem):
        f = pd.read_csv(csv_file, usecols =['cut','direction'])
        utility = 0
        if problem.case == 1:
            solution_properties = pd.DataFrame(columns = ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2'])            # whole solution
            for i, e in f.iterrows():
                cut = problem.df_nodes_dic[e.cut]
                L = [cut.node_info[x] for x in ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']]
                solution_properties.loc[i]=L
                if i == 0:
                    utility = cut.cost_reclaim / cut.cut_tonnage
                else:
                    cost_move = (problem.df_cost[(f.loc[i-1].cut, f.loc[i].cut,
                                                f.loc[i-1].direction, f.loc[i].direction)]['Cost'])
                    cost_reclaim = cut.cost_reclaim
                    cost_total = cost_move + cost_reclaim
                    utility += cost_total / cut.cut_tonnage
            average_properties = np.array(np.mean(solution_properties))
            [u,l]=self.utils.calc_penalty_upper_lower(average_properties)
            violation = u+l
            print(violation,utility)
        elif problem.case == 2:
            solution_properties = {k:pd.DataFrame(columns = ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']) for k in range(problem.demand)}
            parcel = []
            tonnage = 0
            demand = 0
            for i, e in f.iterrows():
                cut = problem.df_nodes_dic[e.cut]
                tonnage += cut.cut_tonnage
                L = [cut.node_info[x] for x in ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']]
                solution_properties[demand].loc[i] = L
                if i == 0:
                    utility = cut.cost_reclaim / cut.cut_tonnage
                else:
                    cost_move = (problem.df_cost[(f.loc[i - 1].cut, f.loc[i].cut,
                                                  f.loc[i - 1].direction, f.loc[i].direction)]['Cost'])
                    cost_reclaim = cut.cost_reclaim
                    cost_total = cost_move + cost_reclaim
                    utility += cost_total / cut.cut_tonnage
                if tonnage >= 1e5*(demand+1):
                    average_properties = np.array(np.mean(solution_properties[demand]))
                    [u, l] = self.utils.calc_penalty_upper_lower(average_properties)
                    violation = u + l
                    parcel.append([violation,utility])
                    if demand+2 <= problem.demand:
                        utility = 0
                        demand += 1
                    # else:
            print(parcel)
            parcel = [np.array(x).astype(float) for x in parcel]
            np.set_printoptions(suppress=True)
            print(sum(parcel))


