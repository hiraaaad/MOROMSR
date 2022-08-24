from core.utils import np, os, pd, Utils, islice, nx
from core.problem import Problem
from core.node import Solution, Cut
from codetiming import Timer
import logging
import copy
import pathlib
import glob
from operator import attrgetter
class manual_alg:
    def __init__(self, problem):
        self.utils = Utils(problem)

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
        self.setup_log()
        self.move_count = 1
        self.pos_reclaimer = None
        self.termination = False
        self.total_capacity = np.round(self.utils.problem.demand * 1e5)
        self.count_local = 0

    def setup_log(self):
        prefix = 'Manual'

        if self.utils.problem.case == 1:
            xx = '/result_case_1/'
            postfix = '_{}%'.format(self.utils.problem.demand * 100)
        else:
            xx = '/result_case_2/'
            postfix = '_{}req'.format(self.utils.problem.demand)


        x = '{}_{}_{}_{}_{}{}'.format(prefix,self.utils.problem.number_stockpile,
                                       self.utils.problem.number_bench,
                                       self.utils.problem.number_cut,
                                       self.utils.problem.demand,
                                       postfix
                                     )



        pathlib.Path(os.getenv("HOME") + xx + x).mkdir(parents=True, exist_ok=True)
        self.log_directory = str(pathlib.Path(os.getenv("HOME") + xx)) + '/' + x

        log_str = '{}/{}'.format(self.log_directory, x)

        self.log_str = log_str + '.log'
        self.log_csv = log_str + '.csv'
        self.log_csv_total = log_str + '_total.csv'

    def reclaim_initial(self, initial_node):
        self.find_next_node_manual(initial_node)
        node = self.utils.problem.df_nodes_dic[initial_node]
        self.solution.obj = node.cost_reclaim / node.cut_tonnage
        self.solution.tonnage_so_far = node.cut_tonnage
        direction = 'SN'
        node_info = node.node_info
        for key in self.solution.visited_info:
            self.solution.visited_info[key].append(node_info[key])
        append_node = (node, direction, self.solution.obj)
        self.solution.visited.append(append_node)
        self.reclaimed_cuts[node.name] = append_node
        self.reclaimed_cuts_keys.append(node.name)
        del self.next_node_list[0]
        self.solution.make_parcel(node)

    def run(self):
        t = Timer("example", text="Time spent: {:.2f}")
        t.start()
        if self.utils.problem.case == 1:
            self.run_case_1()
        else:
            self.run_case_2()
        t.stop()
        time_spent = t.last
        self.report(time_spent)

    def run_case_2(self):
        self.reclaim_initial('01-01-01-01-01')
        for demand in range(self.utils.problem.demand):
            self.termination = False
            while self.termination == False:
                next_node = self.utils.problem.df_nodes_dic[self.next_node_list[0]]
                next_node_direction = 'SN'
                fitness_candidates = self.evaluate_node([(next_node, next_node_direction)])
                next_idx = 0
                self.solution.obj += fitness_candidates[next_idx][0] #cost
                # self.solution.viol = fitness_candidates[next_idx][1] #violation
                self.solution.parcel_list[-1].penalty_mineral_avg = fitness_candidates[next_idx][2]  # average_limits
                self.solution.parcel_list[-1].length += 1
                self.solution.parcel_list[-1].penalty = fitness_candidates[next_idx][1]  # violation
                self.solution.viol = sum([x.penalty for x in self.solution.parcel_list])
                node_info = next_node.node_info
                self.solution.tonnage_so_far += next_node.cut_tonnage

                for key in self.solution.visited_info:
                    self.solution.visited_info[key].append(node_info[key])
                append_node = (next_node, next_node_direction, fitness_candidates[next_idx][0])
                self.solution.visited.append(append_node)
                self.reclaimed_cuts[next_node.name] = append_node
                self.reclaimed_cuts_keys.append(next_node.name)
                self.move_count += 1
                del(self.next_node_list[next_idx])
                # if np.round(self.solution.tonnage_so_far,4) >= np.round(self.total_capacity,4):
                if np.round(self.solution.tonnage_so_far, 4) >= 1e5*(demand+1):
                    self.termination = True
                    self.solution.parcel_list[-1].end = len(self.solution) - 1
                    L = [x[0] for x in self.solution.visited]
                    self.solution.parcel_list[-1].visited = L[self.solution.parcel_list[-1].start:
                                                              self.solution.parcel_list[-1].end + 1]
                    # print(len(self.solution))
                    # self.solution.parcel_list[-1].penalty = self.solution.viol
                    if demand + 2 <= self.utils.problem.demand:
                        self.solution.make_parcel(None)
                    # self.demands.append(copy.deepcopy(self.solution))
                    # self.solution.clean()
        #             self.demands[-1][1] = len(self.solution) - 1
        #             self.demands[-1][2] = self.solution.obj
        #             self.demands[-1][3] = self.solution.viol
        #             self.solution.obj = 0
        # self.solution.obj = sum([x[2] for x in self.demands])
        # self.solution.viol = sum([x[3] for x in self.demands])
        # print(self.solution.obj)
        # print(self.solution.viol)
        print(self.solution.parcel_list)
        print(self.solution)

    def run_case_1(self):
        self.termination = False
        self.total_capacity = np.round(self.utils.problem.demand * self.utils.problem.total_tonnage, 4)
        self.reclaim_initial('01-01-01-01-01')
        while self.termination == False:
            next_node = self.utils.problem.df_nodes_dic[self.next_node_list[0]]
            next_node_direction = 'SN'
            fitness_candidates = self.evaluate_node([(next_node, next_node_direction)])
            next_idx = 0
            self.solution.obj += fitness_candidates[next_idx][0] #cost
            # self.solution.viol = fitness_candidates[next_idx][1] #violation
            self.solution.parcel_list[-1].penalty_mineral_avg = fitness_candidates[next_idx][2]  # average_limits
            self.solution.parcel_list[-1].length += 1
            self.solution.parcel_list[-1].penalty = fitness_candidates[next_idx][1]  # violation
            self.solution.viol = sum([x.penalty for x in self.solution.parcel_list])
            node_info = next_node.node_info
            self.solution.tonnage_so_far += next_node.cut_tonnage

            for key in self.solution.visited_info:
                self.solution.visited_info[key].append(node_info[key])
            append_node = (next_node, next_node_direction, fitness_candidates[next_idx][0])
            self.solution.visited.append(append_node)
            self.reclaimed_cuts[next_node.name] = append_node
            self.reclaimed_cuts_keys.append(next_node.name)
            self.move_count += 1
            del(self.next_node_list[next_idx])
            if np.round(self.solution.tonnage_so_far,4) >= np.round(self.total_capacity,4):
                self.termination = True
                self.solution.parcel_list[-1].end = len(self.solution) - 1
                L = [x[0] for x in self.solution.visited]
                self.solution.parcel_list[-1].visited = L[self.solution.parcel_list[-1].start:
                                                          self.solution.parcel_list[-1].end + 1]


    def find_next_node_manual(self,entry):
        L = []
        for stockpile in range(self.utils.problem.number_stockpile):
            for row in range(self.utils.problem.number_cut):
                for bench in range(self.utils.problem.number_bench):
                    if row+1 != 10:
                        n='01-01-0' + str(stockpile+1) + '-0' + str(bench+1) + '-0' + str(row+1)
                    else:
                        n = '01-01-0' + str(stockpile + 1) + '-0' + str(bench + 1) + '-' + str(row + 1)
                    L.append(n)
        self.next_node_list = L

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
        n = self.solution.parcel_list[-1].length
        if n > 0:
            avg = self.solution.parcel_list[-1].penalty_mineral_avg
            average_limits = avg + ((neighbor_info_properties - avg) / (n + 1))
        else:
            average_limits = neighbor_info_properties
        violation_lower = np.divide(average_limits, self.utils.problem.lower_limits_array,
                                    out=np.ones(average_limits.size, dtype=float) * np.finfo(np.float32).max,
                                    where=self.utils.problem.lower_limits_array != 0) - 1
        penalty_lower = abs(sum([x if x < 0 else 0 for x in violation_lower]))
        violation_upper = (average_limits / self.utils.problem.upper_limits_array) - 1
        penalty_upper = sum([x if x > 0 else 0 for x in violation_upper])
        penalty_neighbor = penalty_lower + penalty_upper + penalty_window

        return (penalty_neighbor, average_limits)

    # def calc_penalty(self, neighbor_info):
    #     visited_temp = {k: [] for k in self.utils.problem.limits_lower}
    #     for k in visited_temp:
    #         visited_temp[k].extend(self.solution.visited_info[k])
    #         visited_temp[k].append(neighbor_info[k])
    #     average_limits = np.array([np.mean(v) for k, v in visited_temp.items() if k in self.utils.problem.limits_lower.keys()],dtype=float)
    #     violation_lower = np.divide(average_limits, self.utils.problem.lower_limits_array,
    #                                 out=np.ones(average_limits.shape, dtype=float)*np.finfo(np.float32).max,
    #                                 where=self.utils.problem.lower_limits_array !=0) - 1
    #     penalty_lower = abs(sum([x if x < 0 else 0 for x in violation_lower]))
    #     violation_upper = (average_limits / self.utils.problem.upper_limits_array) - 1
    #     penalty_upper = sum([x if x > 0 else 0 for x in violation_upper])
    #     penalty_neighbor = penalty_lower + penalty_upper
    #
    #     return penalty_neighbor

    @property
    def pos_current(self):
        index = self.reclaimed_cuts_keys[-1]
        return (index, self.reclaimed_cuts[index][1])
        # return (self.solution.visited[-1][0].name, self.solution.visited[-1][1])
        # return (self.solution.visited_info.iloc[-1].name, self.solution.visited_info.iloc[-1].Direction)

    def report(self,time_spent):

        df_total_csv = pd.DataFrame(columns=['cost',
                                             'violation',
                                             'reclaimed tonnage',
                                             'available tonnage',
                                             'parcel_list',
                                             'seed',
                                             'time',
                                             'count_local', ])
        df_total_csv = df_total_csv.append(
            {'cost': float(self.solution.obj), 'violation': float(self.solution.viol),
             'reclaimed tonnage': float(self.solution.tonnage_so_far),
             'available tonnage': float(self.total_capacity),
             'parcel_list': self.solution.parcel_list, 'seed': self.utils.problem.rng_seed,
             'time': time_spent, 'count_local': self.count_local}, ignore_index=True, )

        df_total_csv = df_total_csv.transpose()

        table_visited = pd.DataFrame(self.solution.visited)
        table_visited.columns = ['Node', 'Direction', 'Cost']
        table_visited = table_visited.set_index('Node')
        table_visited.to_csv(self.log_csv)
        df_total_csv.to_csv(self.log_csv_total)

    # def run(self):
    #     L=[]
    #     self.total_capacity = np.round(self.utils.problem.percentage*self.utils.problem.total_tonnage,4)
    #     self.pos_reclaimer = self.utils.problem.visited_info.iloc[-1].name
    #     # self.pos_reclaimer = self.utils.problem.visited_info.iloc[-1].Cut_ID
    #     # total_cuts = self.utils.problem.number_stockpile * self.utils.problem.number_bench * self.utils.problem.number_cut
    #
    #     # while self.move_count <= total_cuts - 1:
    #     # try:
    #         # while self.tonnage_so_far <= self.utils.problem.percentage*self.utils.problem.total_tonnage:
    #     while self.termination == False:
    #         # with open(self.utils.problem.log_str, "a") as logfile:
    #         #     logfile.write('pos reclaimer is {} \n'.format(self.pos_reclaimer))
    #         neighbors = self.utils.problem.identify_node(next_nodes[0])
    #         neighbors.Direction = 'SN'
    #         (fitness_candidates, candidate_info) = self.utils.evaluate_fitness_manual(neighbors, self.pos_reclaimer)
    #          # only reclamation cost
    #         # with open(self.utils.problem.log_str, "a") as logfile:
    #         #     logfile.write('greedy selection is the node with fitness: {} \n'.format(fitness_candidates[0]))
    #         self.cost_so_far += float(fitness_candidates[0])
    #         self.constraint_violation = float(fitness_candidates[1])
    #         self.tonnage_so_far += float(neighbors.Cut_Tonnage)
    #         L.append(neighbors.Cut_Tonnage)
    #         self.utils.problem.visited_info = self.utils.problem.visited_info.append(neighbors)
    #         self.move_count += 1
    #         self.pos_reclaimer = next_nodes[0]
    #         self.report_csv(next_nodes[0], float(fitness_candidates[0]),
    #                         float(fitness_candidates[2]), neighbors.Cut_Tonnage)
    #         del(next_nodes[0])
    #         # with open(self.utils.problem.log_str, "a") as logfile:
    #         #     logfile.write('new solution string is {} \n'.format(list(self.utils.problem.visited_info.index)))
    #         #     logfile.write('reclaim #{} has been completed \n'.format(self.move_count))
    #         #     logfile.write('\n')
    #         # print('reclaim #{} has been completed \n'.format(self.move_count))
    #         if np.round(self.tonnage_so_far, 4) >= np.round(self.total_capacity, 4):
    #             self.termination = True
    #     # except:
    #     #     print('ERRPR')
    #
    #
    #
    #     # df_total_csv = pd.DataFrame(columns=['cost', 'violation', 'reclaimed tonnage', 'available tonnage'])
    #     # df_total_csv = df_total_csv.append(
    #     #     {'cost': float(self.cost_so_far), 'violation': float(self.constraint_violation),
    #     #      'reclaimed tonnage': float(self.tonnage_so_far),
    #     #      'available tonnage': float(total_capacity)}, ignore_index=True)
    #     # df_total_csv = df_total_csv.transpose()
    #     #
    #     # self.df_csv.to_csv(self.utils.problem.log_csv, index=False)
    #     # df_total_csv.to_csv(self.utils.problem.log_csv_total)
    #
    # def report(self):
    #     # with open(self.utils.problem.log_str, "a") as logfile:
    #     #     logfile.write('Total penalized cost is {} \n'.format(float(self.cost_so_far+self.utils.problem.penalty_coeff*self.constraint_violation)))
    #     #     logfile.write('Total cost is {} \n'.format(float(self.cost_so_far)))
    #     #     logfile.write('Total tonnage is {} \n'.format(float(self.tonnage_so_far)))
    #     #     logfile.write('Total violation is {} \n'.format(float(self.constraint_violation)))
    #     #     logfile.write('Total capacity is {} \n'.format(self.utils.problem.percentage*self.utils.problem.total_tonnage))
    #
    #     df_total_csv = pd.DataFrame(columns=['cost', 'violation', 'reclaimed tonnage', 'available tonnage'])
    #     df_total_csv = df_total_csv.append(
    #         {'cost': float(self.cost_so_far), 'violation': float(self.constraint_violation),
    #          'reclaimed tonnage': float(self.tonnage_so_far),
    #          'available tonnage': float(self.total_capacity)}, ignore_index=True)
    #     df_total_csv = df_total_csv.transpose()
    #
    #     self.df_csv.to_csv(self.utils.problem.log_csv, index=False)
    #     df_total_csv.to_csv(self.utils.problem.log_csv_total)

    def local_operator(self, solution, i, j):
        if self.problem.ls == 1:  # swap
            return self.swap_pos(solution, i, j)
        elif self.problem.ls == 2:  # insert
            return self.insert_pos(solution, i, j)
        elif self.problem.ls == 3:  # inverse
            return self.inverse_pos(solution, i, j)

    # def reclaimed_before(self, node, before, index):
    #     return node in before[:index]


    def valid_local(self, segment, local_solution):
        condition = True
        L=[]
        for edge in segment:
            index = edge[1]
            cut = local_solution.visited[index][0]
            direction = local_solution.visited[edge[1]][1]
            lst = cut.prec[direction]
            reclaimer_name = local_solution.reclaimed_cuts_keys[edge[0]]
            cut_name = local_solution.reclaimed_cuts_keys[edge[1]]
            if len(lst) > 0:
                if self.utils.problem.identify_node_stockpile(cut_name) == self.utils.problem.identify_node_stockpile(reclaimer_name):
                    if (reclaimer_name,cut_name) not in self.utils.problem.G.edges:
                        return False
                else:
                    before = local_solution.reclaimed_cuts_keys[:index]
                    for node in list(lst.Job_Step_1):
                        if node not in before:
                            return False
            # before = X[:index]
            #
            # before_cuts = [x[0].name for x in local_solution.visited[:index]]
            # before_cuts = map(x.name, local_solution.visited[:index][0])
            # before_cuts = np.array([x[0].name for x in local_solution.visited[:index]])
            # before = set(local_solution.reclaimed_cuts_keys[:index])
            # before = local_solution.reclaimed_cuts_keys[:index]
            # if len(lst) > 0:
            #
        return condition

    # def valid_local(self, segment, solution):
    #     # t = Timer()
    #     # t.start()
    #     # if i == 2 and j == 4:
    #     #     print('X')
    #     segment_idx = [x[0].name for x in segment]
    #     # print(len(segment) - 1)
    #     for idx in range(len(segment) - 1):
    #         direction = solution[idx][1]
    #         lst = segment[idx][0].prec[direction]
    #         if len(lst) > 0:
    #             2+2
    #             # for node in list(lst.Job_Step_1):
    #             #     2+2
    #                 # if node in segment_idx:
    #                     # node
    #         #             if segment_idx.index(node) > idx:
    #         #                 # t.stop()
    #         #                 return False
    #     # t.stop()
    #     return True



            # for node in list(lst.Job_Step_1):



        #     pos_reclaimer = local_solution.visited[edge[0]][0].name
        #     neighbor_name = local_solution.visited[edge[1]][0].name
        #     direction_reclaimer = local_solution.visited[edge[0]][1]
        #     neighbor_direction = local_solution.visited[edge[0]][1]
        #     L.append([pos_reclaimer, neighbor_name, direction_reclaimer, neighbor_direction])
        # print(len(segment))
        # # # t = Timer()
        # # # t.start()
        # # # if i == 2 and j == 4:
        # # #     print('X')
        # segment_idx = [x[0].name for x in segment]
        # # print(len(segment) - 1)
        # for idx in range(len(segment) - 1):
        #     direction = solution[idx][1]
        #     lst = segment[idx][0].prec[direction]
        #     if len(lst) > 0:
        #         for node in list(lst.Job_Step_1):
        #             if node in segment_idx:
        #                 if segment_idx.index(node) > idx:
        #                     # t.stop()
        #                     return False
        # # t.stop()
        # return True

    def calc_cost_edge(self, edge, local_solution):
        pos_reclaimer = local_solution.visited[edge[0]][0].name
        neighbor_name = local_solution.visited[edge[1]][0].name
        direction_reclaimer = local_solution.visited[edge[0]][1]
        neighbor_direction = local_solution.visited[edge[0]][1]
        cost_move = self.problem.df_cost[(pos_reclaimer, neighbor_name,
                                          direction_reclaimer, neighbor_direction)]['Cost']
        cost_reclaim = local_solution.visited[edge[1]][0].node_info['Cost']
        cost_total = cost_move + cost_reclaim
        cost_neighbor = cost_total / local_solution.visited[edge[1]][0].node_info[
            'Cut_Tonnage']  # + self.problem.penalty_coeff*penalty_neighbor
        cost_neighbor = round(cost_neighbor, 6)
        return cost_neighbor

    def swap_pos(self, solution, i, j):
        S = Solution()
        L = solution.visited
        X = solution.reclaimed_cuts_keys
        L[j], L[i] = L[i], L[j]
        X[j], X[i] = X[i], X[j]
        S.visited = L
        S.reclaimed_cuts_keys = X
        return S

    def insert_pos(self, solution, i, j):
        S = Solution()
        L = solution.visited
        L.insert(i + 1, L[j])
        where_idx = [idx for idx, e in enumerate(L) if e == L[i + 1]]
        del L[where_idx[1]]
        S.visited = L
        return S

    def inverse_pos(self, solution, i, j):
        S = Solution()
        L = solution.visited
        L = L[:i] + L[i:j + 1][::-1] + L[j + 1:]
        S.visited = L
        return S

    def identify_changed_edges(self, local_solution, i, j):
        L=[]
        if self.problem.ls == 1:
            L = [(i - 1, i), (j - 1, j), (i, i + 1)]
            if j + 1 < len(local_solution.visited):
                L.append((j, j + 1))
        elif self.problem.ls == 2:
            L = []
            index = i
            while index < j - 1:
                L.append((index, index + 1))
                index += 1
            if j + 1 < len(local_solution.visited):
                L.append((j, j + 1))
        elif self.problem.ls == 3:
            index = i - 1
            L = []
            while index < j:
                L.append((index, index + 1))
                index += 1
            if j + 1 < len(local_solution.visited):
                L.append((j, j + 1))
        L = set(L)
        return L

    def calc_cost_local(self, local_solution, segment):
        for edge in segment:
            cost_neighbor = self.calc_cost_edge(edge, local_solution)
            local_solution.visited[edge[1]] = (
            local_solution.visited[edge[1]][0], local_solution.visited[edge[1]][1], cost_neighbor)
        local_solution.obj = sum([x[2] for x in local_solution.visited])
        return local_solution


    # def calc_cost_swap(self, local_solution, i, j):
    #     # we should find at most 4 unique affected edges
    #     L = [(i - 1, i), (j - 1, j), (i, i + 1)]
    #     if j + 1 < len(local_solution.visited):
    #         L.append((j, j + 1))
    #     L = set(L)
    #     for edge in L:
    #         cost_neighbor = self.calc_cost_edge(edge, local_solution)
    #         # replace cost
    #         local_solution.visited[edge[1]] = (
    #         local_solution.visited[edge[1]][0], local_solution.visited[edge[1]][1], cost_neighbor)
    #     return local_solution
    #
    # def calc_cost_insert(self, local_solution, i, j):
    #     index = i
    #     L = []
    #     while index < j - 1:
    #         L.append((index, index + 1))
    #         index += 1
    #     if j + 1 < len(local_solution.visited):
    #         L.append((j, j + 1))
    #     L = set(L)
    #     for edge in L:
    #         cost_neighbor = self.calc_cost_edge(edge, local_solution)
    #         local_solution.visited[edge[1]] = (
    #         local_solution.visited[edge[1]][0], local_solution.visited[edge[1]][1], cost_neighbor)
    #     return local_solution
    #
    # def calc_cost_inverse(self, local_solution, i, j):
    #     index = i - 1
    #     L = []
    #     while index < j:
    #         L.append((index, index + 1))
    #         index += 1
    #     if j + 1 < len(local_solution.visited):
    #         L.append((j, j + 1))
    #     L = set(L)
    #     for edge in L:
    #         cost_neighbor = self.calc_cost_edge(edge, local_solution)
    #         # replace cost
    #         local_solution.visited[edge[1]] = (
    #         local_solution.visited[edge[1]][0], local_solution.visited[edge[1]][1], cost_neighbor)
    #     return local_solution

    def calc_penalty_local(self, local_solution, initial_solution):
        # initial = [x[0] for x in initial_solution.visited]
        interval_list = [(x.start, x.end) for x in initial_solution.parcel_list]
        new = [x[0] for x in local_solution.visited]
        new_parcel = [new[interval[0]:interval[1] + 1] for interval in interval_list]
        initial_parcel = [parcel.visited for parcel in initial_solution.parcel_list]
        number_parcel = len(initial_parcel)
        local_solution.generate_parcel(initial_solution, new_parcel)
        new_parcel_avg_penalty = [x.penalty_mineral_avg for x in local_solution.parcel_list]
        for id in range(number_parcel):
            New = list(set(new_parcel[id]) - set(initial_parcel[id]))
            Old = list(set(initial_parcel[id]) - set(new_parcel[id]))
            if len(New) > 0 and len(Old) > 0:
                # print(New, Old)
                for cut_number in range(len(New)):
                    a = np.array([Old[cut_number].node_info[x] for x in ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']])
                    b = np.array([New[cut_number].node_info[x] for x in ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']])
                    n = len(initial_parcel[id])
                    new_parcel_avg_penalty[id] += (b - a) / n
                    (penalty_lower, penalty_upper) = self.calc_penalty_upper_lower(new_parcel_avg_penalty[id])
                    penalty_parcel = penalty_lower + penalty_upper
                    # if penalty_parcel < local_solution.parcel_list[id].penalty:
                    #     print('X')
                    local_solution.parcel_list[id].penalty = penalty_parcel
                    local_solution.parcel_list[id].penalty_mineral_avg = new_parcel_avg_penalty[id]
                    local_solution.viol = sum(x.penalty for x in local_solution.parcel_list)

        return local_solution

    def calc_penalty_upper_lower(self,input):
        violation_lower = np.divide(input, self.problem.lower_limits_array,
                                    out=np.ones(input.size, dtype=float) * np.finfo(
                                        np.float32).max,
                                    where=self.problem.lower_limits_array != 0) - 1
        penalty_lower = abs(sum([x if x < 0 else 0 for x in violation_lower]))
        violation_upper = (input / self.problem.upper_limits_array) - 1
        penalty_upper = sum([x if x > 0 else 0 for x in violation_upper])
        return penalty_lower, penalty_upper

    # def local_search(self, initial_solution):
    #     S = []
    #     t = Timer("example", text="Time spent: {:.2f}", logger=logging.warning)
    #     initial_visited = initial_solution.visited
    #     X=[]
    #     t.start()
    #     for i in range(1, len(initial_visited)):
    #         for j in range(1, len(initial_visited)):
    #             if j > i:
    #     #             # print(i, j)
    #                 local_solution = self.local_operator(initial_solution, i, j) #Numba?
    #                 local_solution.viol = initial_solution.viol
    #                 segment = self.identify_changed_edges(local_solution, i, j) #Numba?
    #                 # local_solution.reclaimed_cuts_keys = [x[0].name for x in local_solution.visited]
    #     # local_solution.reclaimed_cuts_keys = [x[0].name for x in local_solution.visited[:max(max(segment)) + 1]]
    #                 self.valid_local(segment, local_solution):
    #     #                 local_solution = self.calc_cost_local(local_solution, segment)
    #     #                 local_solution = self.calc_penalty_local(local_solution, initial_solution)
    #     #                 S.append((i, j, local_solution))
    #     # # t.stop()
    #     # #             if self.valid_local(segment, initial_visited):
    #     # #
    #     # #
    #     # L = [x[2] for x in S]
    #     # cost = [x.obj for x in L]
    #     # v = [x.viol for x in L]
    #     # rank = np.lexsort((cost, v))
    #     t.stop()
    #     # return L[rank[0]]
    #     # t.stop()
    #     # print(X)
    #     # print(max(X))
    #     # print(sum(X))
    #     # print(len(X))
    #     # print(len(self.solution))
    #     # print(sum(X))

    # def local_search_iterative(self):
    #     self.problem = self.utils.problem
    #     initial_solution = self.solution
    #     termination = False
    #     X = []
    #     while termination is False:
    #         # t = Timer("example", text="Time spent: {:.2f}", logger=logging.warning)
    #         # t = Timer()
    #     #             # t.start()
    #         local_solution = self.utils.local_search(initial_solution)
    #     #             # t.stop()
    #     #             # X.append(t.last)
    #         print(local_solution)
    #         if local_solution < initial_solution:
    #             print(local_solution, initial_solution)
    #             initial_solution = local_solution
    #         else:
    #             termination = True
    #     self.solution = initial_solution
    #     # t = Timer()
    #     # t.start()
    #     # t.stop()
    #     return local_solution