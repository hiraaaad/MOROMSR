from core.utils import np, os, pd, Utils, islice
from core.problem import Problem
from core.node import Solution
from codetiming import Timer
import logging
import copy
import pathlib
import glob
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

        self.solution = Solution()
        self.reclaimed_cuts = {}
        self.reclaimed_cuts_keys = []
        self.df_nodes = problem.df_nodes
        self.setup_log()
        self.move_count = 1
        self.pos_reclaimer = None
        self.termination = False
        self.total_capacity = np.round(self.utils.problem.demand * 1e5)

    def setup_log(self):
        prefix = 'Manual'

        x = '{}_{}_{}_{}_{}_req'.format(prefix,self.utils.problem.number_stockpile,
                                       self.utils.problem.number_bench,
                                       self.utils.problem.number_cut,
                                       self.utils.problem.demand,
                                     )
        xx = '/result_new/'
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

    def report(self):

        df_total_csv = pd.DataFrame(columns=['cost', 'violation', 'reclaimed tonnage', 'available tonnage'])
        df_total_csv = df_total_csv.append(
            {'cost': float(self.solution.obj), 'violation': float(self.solution.viol),
             'reclaimed tonnage': float(self.solution.tonnage_so_far),
             'available tonnage': float(self.total_capacity)}, ignore_index=True)

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