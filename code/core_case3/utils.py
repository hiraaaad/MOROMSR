import os
import glob
import pandas as pd
import re
import copy
import networkx as nx
from codetiming import Timer
import logging
import numpy as np
from itertools import islice
from operator import attrgetter
from core_case3.node import Solution
# seed

class Utils:
    def __init__(self, problem):
        self.problem = problem
        if self.problem.greedy_alpha > 0:
            np.random.seed(self.problem.rng_seed)

    # def find_neighborhood(self, direction):
    #     # find neighbors in the current stockpile & add the accessible nodes to the end
    #     lst = []
    #     (pos_reclaimer, direction_reclaimer) = self.problem.pos_current
    #     neighbor_list = self.problem.df_nodes_dic[pos_reclaimer].prec_1[direction_reclaimer]
    #     if len(neighbor_list) > 0:
    #         lst_accssible_nodes = list(self.problem.accessible_nodes.index)
    #         for node in list(neighbor_list.Job_Step_2):
    #             if not self.problem.reclaimed(node) and node not in lst_accssible_nodes:
    #                 prec = self.problem.df_nodes_dic[node].prec[direction]
    #                 # prec = self.problem.df_prec.loc[node, direction]
    #                 if len(prec)>0:
    #                     if all([x in self.solution.visited_info.index for x in list(prec.Job_Step_1)]):
    #                         lst.append(node)
    #     output=[]
    #     for node in lst:
    #         output.append((node,direction))
    #     return output

    def calc_penalty(self, visited_temp):
        average_limits = visited_temp[self.problem.limits.keys()].mean()
        comparison = average_limits <= np.array(list(self.problem.limits.values()))
        violation = []
        for i, e in comparison.iteritems():
            # if e is False:
                # normalization
            violation.append(average_limits[i] / self.problem.limits[i] - 1)
            #else:
            #     violation.append(0)
        penalty_neighbor = sum([x if x > 0 else 0 for x in violation])
        if penalty_neighbor > 0:
            penalty_report = penalty_neighbor
        else:
            penalty_report = sum(violation)
        # with open(self.problem.log_str, "a") as logfile:
        #     logfile.write('constraint violation for cut {} w/ direction {}: {} | total violation is {} \n'.format(
        #         visited_temp.iloc[-1].name, visited_temp.iloc[-1].Direction, sum(violation), violation))
        return (penalty_neighbor,penalty_report)

    def evaluate_fitness_manual(self,neighbors,pos_reclaimer):
        cost_candidate = []
        candidate_info = pd.DataFrame(columns=self.problem.columns)
        visited_temp = self.problem.visited_info.copy()
        visited_temp = visited_temp.append(neighbors)
        cost_move = self.problem.df_cost.loc[(pos_reclaimer, neighbors.name,
                                              'SN',
                                              neighbors.Direction), 'Cost']
        # cost_move = self.problem.df_cost.loc[(pos_reclaimer, neighbors.name,
        #                                       self.problem.visited_info.iloc[-1].Direction,
        #                                       neighbors.Direction), 'Cost']
        # for two directions, right now we only work on SN
        cost_reclaim = neighbors.Cost
        cost_total = cost_move + cost_reclaim
        (penalty_neighbor,penalty_report) = self.calc_penalty(visited_temp)
        cost_neighbor = cost_total / neighbors.Cut_Tonnage # + self.problem.penalty_coeff * penalty_neighbor
        cost_neighbor = round(cost_neighbor, 6)
        penalty_neighbor = round(penalty_neighbor, 6)
        penalty_report = round(penalty_report, 6)
        fitness_candidates = (float(cost_neighbor), penalty_neighbor, penalty_report)
        candidate_info = candidate_info.append(neighbors)
        # cost_candidate.append((cost_total,penalty_neighbor))
        return (fitness_candidates, candidate_info)

    # def greedy_selection(self,fitness):
    #     # this function returns the index in the fitness vector which greedy algorithm selects
    #     candidate = fitness[0]
    #     for f in fitness[1:]:
    #         if self.problem.constraint_type == 0:
    #             # compare f with next candidate
    #             if candidate[1] == 0 and f[1] == 0:  # both feasible
    #                 winner_idx = np.array([candidate[0], f[0]]).argmin()
    #             elif candidate[1] > 0 and f[1] > 0:  # both infeasible
    #                 winner_idx = np.array([candidate[1], f[1]]).argmin()
    #             else:
    #                 winner_idx = np.array([candidate[1], f[1]]).argmin()
    #         else: # unc
    #             winner_idx = np.array([candidate[0], f[0]]).argmin()
    #
    #         if winner_idx != 0:
    #             candidate = f
    #     winner_idx = fitness.index(candidate)
    #     with open(self.problem.log_str, "a") as logfile:
    #         logfile.write('greedy selection among {} is the node with fitness: {} with index of {} \n'.format(fitness, candidate,
    #                                                                                            winner_idx))
    #
    #     # print('greedy selection among {} is the node with fitness: {} with index of {}'.format(fitness, candidate,
    #     #                                                                                        winner_idx))
    #     return winner_idx

    def rwheel(self,fitness):
        s = [(1 / i) ** self.problem.greedy_alpha for i in fitness]
        ss = [i / sum(s) for i in s]
        csum = np.cumsum(ss)
        rand = np.random.random()
        # print(rand)
        for i,e in enumerate(csum):
            if rand <= e:
                idx = i
                return idx

    def mynormal(self, value, m, M, factor):
        if value != 0:
            if M - m == 0:
                obj = 0 + factor
            else:
                obj = 0 + (((value - m) * (1 - 0)) / (M - m)) + factor
            return obj
        else:
            return 0

    def calc_normal_cluster(self, cluster):
        X = []
        lst_penalty_main = [i[0][1] for i in cluster if i[0][1] > 0]
        lst_penalty_win = [i[0][4] for i in cluster if i[0][4] > 0]
        lst_cost = [i[0][0] for i in cluster if i[0][0] > 0]
        for solution in cluster:
            obj_cost = self.mynormal(solution[0][0], min(lst_cost), max(lst_cost), 1)
            if len(lst_penalty_main) > 0:
                obj_penalty_main = self.mynormal(solution[0][1], min(lst_penalty_main), max(lst_penalty_main), 100)
            else:
                obj_penalty_main = 0

            if len(lst_penalty_win) > 0:
                obj_penalty_win = self.mynormal(solution[0][4], min(lst_penalty_win), max(lst_penalty_win), 10)
            else:
                obj_penalty_win = 0

            obj = obj_cost + obj_penalty_main + obj_penalty_win
            X.append(obj)
        return X


    def greedy_selection_constrained_case3(self,fitness):
        if self.problem.greedy_alpha == 0:
            cost = np.array([x[0] for x in fitness])
            penalty_main = [x[1] for x in fitness]
            penalty_win = [x[4] for x in fitness]
            r = np.lexsort((cost, penalty_win, penalty_main))
            winner_idx = r[0]
        else:
            'FIND CLUSTER'
            Feasible = []
            SemiFeasible = []
            InFeasible = []
            for idx, i in enumerate(fitness):
                if i[1] == 0 and i[4] == 0:
                    Feasible.append((i, idx))
                elif i[1] != 0:
                    InFeasible.append((i, idx))
                elif i[4] > 0:
                    SemiFeasible.append((i, idx))
                else:
                    print('XX')
            X = []
            mapping = []
            for cluster in [Feasible, InFeasible, SemiFeasible]:
                if len(cluster) > 0:
                    X.extend(self.calc_normal_cluster(cluster))
                    mapping.append([i[1] for i in cluster])

            fitness_ext = []
            fitness_ext.extend(X)

            map_ext = []
            for i in mapping:
                map_ext.extend(i)

            winner_idx = map_ext[self.rwheel(fitness_ext)]

        return winner_idx

        # cost = np.array([x[0] for x in fitness])
        # penalty_main = [x[1] for x in fitness]
        # penalty_win = [x[4] for x in fitness]
        # r = np.lexsort((cost, penalty_win, penalty_main))
        # if self.problem.greedy_alpha == 0:
        #     winner_idx = r[0]
        # else:
        #     ff = self.mynormal(np.array(penalty_main), 'penalty_main') + self.mynormal(np.array(penalty_win), 'penalty_win') \
        #          + self.mynormal(cost, 'cost')
        #
        #     winner_idx = self.rwheel(ff)
        # return winner_idx
    # def greedy_selection_constrained_case3(self,fitness):
    #     cost = np.array([x[0] for x in fitness])
    #     penalty_main = [x[1] for x in fitness]
    #     penalty_win = [x[4] for x in fitness]
    #     r = np.lexsort((cost, penalty_win, penalty_main))
    #
    #     if self.problem.greedy_alpha ==0:
    #         # winner_idx = self.rwheel(ff)
    #         winner_idx = r[0]
    #
    #     else:
    #         penalty_collective = np.array(penalty_main) + np.array(penalty_win)
    #         all_feasible = np.where(penalty_collective == 0)[0]
    #         if len(all_feasible) == len(fitness):
    #             winner_idx = self.rwheel(cost)
    #         else:
    #             if len(all_feasible)>0:
    #                 worst_feasible = max(cost[all_feasible])
    #             else:
    #                 worst_feasible = 1
    #             ff = []
    #             for i, e in enumerate(cost):
    #                 if i in all_feasible:
    #                     ff.append(e)
    #                 else:
    #                     ff.append(1.5 * worst_feasible + int(np.array(np.where(r==i))))
    #
    #             winner_idx = self.rwheel(ff)
    #
    #     return winner_idx


    def assign_node_in_stockpile_dic(self,neighbors,next_node_info):
        # this function returns a dictionary with all the nodes in the df:
        output = {k: [] for k in range(self.problem.number_stockpile)}
        # first remove next_node from neighbors
        next_node_stockpile = self.problem.identify_node_stockpile(next_node_info.Cut_ID)
        neighbors[next_node_stockpile] = neighbors[next_node_stockpile][
            neighbors[next_node_stockpile].Cut_ID != next_node_info.Cut_ID]
        #
        for stockpile in neighbors:
            for i,e in stockpile.iterrows():
                node_name = e.Cut_ID
                node_stockpile = self.problem.identify_node_stockpile(node_name)
                output[node_stockpile].append(node_name)
        return output

    def find_best_idx(self,df):
        min_violation = df.loc[df['1'] == df['1'].min()]
        if len(min_violation) == 0:
            return min_violation.iloc[0].name
        else:
            return min_violation['0'].idxmin()
    ## DEPRECATED
    # def local_search(self,initial_solution):
    #     S=[]
    #     t = Timer("example", text="Time spent: {:.2f}", logger=logging.warning)
    #     initial_visited = initial_solution.visited
    #     for i in range(1, len(initial_visited)):
    #         for j in range(1, len(initial_visited)):
    #             if j > i:
    #                 # print(i, j)
    #                 local_solution = self.local_operator(initial_visited,i,j)
    #                 local_solution.viol = initial_solution.viol
    #                 start = i
    #                 end = min(j + 1, len(local_solution))
    #                 segment = local_solution.segment(start,end)
    #                 if self.valid_local(segment, initial_visited):
    #                     local_solution = self.calc_cost_local(local_solution, i, j)
    #                     # local_solution.parcel_list = initial_solution.parcel_list
    #                     local_solution = self.calc_penalty_local(local_solution, initial_solution)
    #                     # local_solution.viol = sum([x.penalty for x in local_solution.parcel_list])
    #                     local_solution.obj = sum([x[2] for x in local_solution.visited])
    #
    #                     # print(local_solution)
    #                     # if local_solution < initial_solution:
    #                     #     return local_solution
    #     # return local_solution
    #     #                 print(initial_solution)
    #     #                 print(initial_solution.parcel_list)
    #     #                 print(local_solution)
    #     #                 print(local_solution.parcel_list)
    #                     S.append((i,j,local_solution))
    #     L = [x[2] for x in S]
    #     cost = [x.obj for x in L]
    #     v = [x.viol for x in L]
    #     rank = np.lexsort((cost, v))
    #     return L[rank[0]]
    #         # print(min(L, key=attrgetter('viol')))
    #         # return min(L, key=attrgetter('viol'))
    #
    # def local_operator(self, solution,i,j):
    #     if self.problem.ls == 1: #swap
    #         return self.swap_pos(solution, i, j)
    #     elif self.problem.ls == 2: #insert
    #         return self.insert_pos(solution, i, j)
    #     elif self.problem.ls == 3: #inverse
    #         return self.inverse_pos(solution, i, j)
    #
    # def calc_cost_local(self, local_solution, i, j):
    #
    #     if self.problem.ls == 1: #swap
    #         return self.calc_cost_swap(local_solution, i, j)
    #     elif self.problem.ls == 2: #insert
    #         return  self.calc_cost_insert(local_solution, i, j)
    #     elif self.problem.ls == 3: #inverse
    #         return self.calc_cost_inverse(local_solution, i, j)
    #
    # def valid_local(self,segment, solution):
    #     # if i == 2 and j == 4:
    #     #     print('X')
    #     segment_idx = [x[0].name for x in segment]
    #     for idx in range(len(segment)-1):
    #         direction = solution[idx][1]
    #         lst = segment[idx][0].prec[direction]
    #         if len(lst)>0:
    #             for node in list(lst.Job_Step_1):
    #                 if node in segment_idx:
    #                     if segment_idx.index(node) > idx:
    #                         return False
    #     return True
    #
    # def calc_cost_edge(self, edge, local_solution):
    #     pos_reclaimer = local_solution.visited[edge[0]][0].name
    #     neighbor_name = local_solution.visited[edge[1]][0].name
    #     direction_reclaimer = local_solution.visited[edge[0]][1]
    #     neighbor_direction = local_solution.visited[edge[0]][1]
    #     cost_move = self.problem.df_cost[(pos_reclaimer, neighbor_name,
    #                                       direction_reclaimer, neighbor_direction)]['Cost']
    #     cost_reclaim = local_solution.visited[edge[1]][0].node_info['Cost']
    #     cost_total = cost_move + cost_reclaim
    #     cost_neighbor = cost_total / local_solution.visited[edge[1]][0].node_info[
    #         'Cut_Tonnage']  # + self.problem.penalty_coeff*penalty_neighbor
    #     cost_neighbor = round(cost_neighbor, 6)
    #     return cost_neighbor
    #
    # def swap_pos(self, solution, i, j):
    #     S = Solution()
    #     L = [x for x in solution]
    #     L[j], L[i] = L[i], L[j]
    #     S.visited = L
    #     return S
    #
    # def insert_pos(self, solution, i, j):
    #     S = Solution()
    #     L = [x for x in solution]
    #     L.insert(i + 1, L[j])
    #     where_idx = [idx for idx, e in enumerate(L) if e == L[i + 1]]
    #     del L[where_idx[1]]
    #     S.visited = L
    #     return S
    #
    # def inverse_pos(self, solution, i, j):
    #     S = Solution()
    #     L = [x for x in solution]
    #     L = L[:i] + L[i:j + 1][::-1] + L[j + 1:]
    #     S.visited = L
    #     return S
    #
    # def calc_cost_swap(self, local_solution, i, j):
    #     # we should find at most 4 unique affected edges
    #     L = [(i - 1, i), (j - 1, j), (i, i + 1)]
    #     if j+1 < len(local_solution.visited):
    #         L.append((j, j + 1))
    #     L = set(L)
    #     for edge in L:
    #         cost_neighbor = self.calc_cost_edge(edge, local_solution)
    #         # replace cost
    #         local_solution.visited[edge[1]] = (local_solution.visited[edge[1]][0], local_solution.visited[edge[1]][1], cost_neighbor)
    #     return local_solution
    #
    # def calc_cost_insert(self, local_solution, i, j):
    #     index = i
    #     L = []
    #     while index < j-1:
    #         L.append((index, index+1))
    #         index += 1
    #     if j + 1 < len(local_solution.visited):
    #         L.append((j, j + 1))
    #     L = set(L)
    #     for edge in L:
    #         cost_neighbor = self.calc_cost_edge(edge, local_solution)
    #         local_solution.visited[edge[1]] = (local_solution.visited[edge[1]][0], local_solution.visited[edge[1]][1], cost_neighbor)
    #     return local_solution
    #
    # def calc_cost_inverse(self, local_solution, i, j):
    #     index = i-1
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
    #         local_solution.visited[edge[1]] = (local_solution.visited[edge[1]][0], local_solution.visited[edge[1]][1], cost_neighbor)
    #     return local_solution

    # def calc_penalty_local(self, local_solution, initial_solution):
    #     # initial = [x[0] for x in initial_solution.visited]
    #     interval_list = [(x.start,x.end) for x in initial_solution.parcel_list]
    #     new = [x[0] for x in local_solution.visited]
    #     new_parcel= [new[interval[0]:interval[1]+1] for interval in interval_list]
    #     initial_parcel= [parcel.visited for parcel in initial_solution.parcel_list]
    #     number_parcel = len(initial_parcel)
    #     local_solution.generate_parcel(initial_solution, new_parcel)
    #     new_parcel_avg_penalty = [x.penalty_mineral_avg for x in local_solution.parcel_list]
    #     for id in range(number_parcel):
    #         New = list(set(new_parcel[id]) - set(initial_parcel[id]))
    #         Old = list(set(initial_parcel[id]) - set(new_parcel[id]))
    #         if len(New)>0 and len(Old)>0:
    #             # print(New, Old)
    #             for cut_number in range(len(New)):
    #                 a = np.array([Old[cut_number].node_info[x] for x in ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']])
    #                 b = np.array([New[cut_number].node_info[x] for x in ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']])
    #                 n = len(initial_parcel[id])
    #                 new_parcel_avg_penalty[id] += (b-a)/n
    #                 (penalty_lower, penalty_upper) = self.calc_penalty_upper_lower(new_parcel_avg_penalty[id])
    #                 penalty_parcel = penalty_lower + penalty_upper
    #                 # if penalty_parcel < local_solution.parcel_list[id].penalty:
    #                 #     print('X')
    #                 local_solution.parcel_list[id].penalty = penalty_parcel
    #                 local_solution.parcel_list[id].penalty_mineral_avg = new_parcel_avg_penalty[id]
    #                 local_solution.viol = sum(x.penalty for x in local_solution.parcel_list)
    #
    #     return local_solution
    #
    #
    #
    # def calc_penalty_upper_lower(self,input):
    #     violation_lower = np.divide(input, self.problem.lower_limits_array,
    #                                 out=np.ones(input.size, dtype=float) * np.finfo(
    #                                     np.float32).max,
    #                                 where=self.problem.lower_limits_array != 0) - 1
    #     penalty_lower = abs(sum([x if x < 0 else 0 for x in violation_lower]))
    #     violation_upper = (input / self.problem.upper_limits_array) - 1
    #     penalty_upper = sum([x if x > 0 else 0 for x in violation_upper])
    #     return penalty_lower, penalty_upper
    ## DEPRECATED
    def local_operator(self, solution, i, j):
        condition = True
        if self.problem.ls == 1 and j>i:  # swap
            return (self.swap_pos(solution, i, j), condition)
        elif self.problem.ls == 2:  # insert
            return (self.insert_pos(solution, i, j), condition)
        elif self.problem.ls == 3 and j>i:  # inverse
            return (self.inverse_pos(solution, i, j), condition)
        else:
            condition = False
            return (solution, condition)

    def valid_local(self, segment, local_solution, i, j):
        if self.problem.ls == 1 or self.problem.ls == 3:
            piece = local_solution.visited[i:j + 1]
            idx = i
            for node in piece:
                cut = node[0]
                direction = node[1]
                lst = cut.prec[direction]
                if len(lst) > 0:
                    before = set(local_solution.reclaimed_cuts_keys[:idx])
                    if not all([x in before for x in list(lst.Job_Step_1)]):
                        return False
                idx += 1
            return True
        elif self.problem.ls == 2:
            lst = local_solution.visited[i][0].prec[local_solution.visited[i][1]]
            if len(lst) > 0:
                cut_1_prec = list(lst.Job_Step_1)
                for cut in cut_1_prec:
                    if cut not in local_solution.reclaimed_cuts_keys[:i]:
                        return False
            lst = local_solution.visited[i][0].prec_1[local_solution.visited[i][1]]
            if len(lst) > 0:
                cut_2_prec = list(lst.Job_Step_2)
                for cut in cut_2_prec:
                    if cut in local_solution.reclaimed_cuts_keys[:i]:
                        return False
            return True

        #
        #
        # condition = True
        # L=[]
        # for edge in segment:
        #     index = edge[1]
        #     cut = local_solution.visited[index][0]
        #     direction = local_solution.visited[edge[1]][1]
        #     # reclaimer_name = local_solution.reclaimed_cuts_keys[edge[0]]
        #     # cut_name = local_solution.reclaimed_cuts_keys[edge[1]]
        #     # if self.problem.identify_node_stockpile(cut_name) == self.problem.identify_node_stockpile(reclaimer_name):
        #     #     if (reclaimer_name, cut_name) not in self.problem.G.edges:
        #     #         return False
        #     # else:
        #     lst = cut.prec[direction]
        #     if len(lst) > 0:
        #             before = set(local_solution.reclaimed_cuts_keys[:index])
        #             return all([x in before for x in list(lst.Job_Step_1)])
        #             # for node in list(lst.Job_Step_1):
        #             #     if node not in before:
        #             #         return False
        # return condition

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
        S = Solution(solution)
        S.visited[j], S.visited[i] = S.visited[i], S.visited[j]
        S.reclaimed_cuts_keys[j], S.reclaimed_cuts_keys[i] = S.reclaimed_cuts_keys[i], S.reclaimed_cuts_keys[j]
        return S

    def insert_pos(self, solution, i, j):
        S = Solution(solution)
        value_1 = S.visited[j]
        value_2 = S.reclaimed_cuts_keys[j]
        del S.visited[j]
        del S.reclaimed_cuts_keys[j]
        S.visited.insert(i, value_1)
        S.reclaimed_cuts_keys.insert(i, value_2)
        return S

    def inverse_pos(self, solution, i, j):
        S = Solution(solution)
        S.visited = S.visited[:i] + S.visited[i:j + 1][::-1] + S.visited[j + 1:]
        S.reclaimed_cuts_keys = S.reclaimed_cuts_keys[:i] + S.reclaimed_cuts_keys[i:j + 1][::-1] + S.reclaimed_cuts_keys[j + 1:]
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

    def calc_cost_local(self, local_solution, L):
        # t = Timer("example", text="Time spent: {:.2f}", logger=logging.warning)
        # t.start()
        edges = [(x[0] - 1, x[0]) for x in L]
        for edge in edges:
            cost_neighbor = self.calc_cost_edge(edge, local_solution)
            local_solution.visited[edge[1]] = (
                local_solution.visited[edge[1]][0], local_solution.visited[edge[1]][1], cost_neighbor)
        local_solution.obj = sum([x[2] for x in local_solution.visited])
        return local_solution

    def calc_penalty_window_local(self, local_solution):
        penalty_window = []
        for parcel in local_solution.parcel_list:
            L = []
            parcel_penalty = {k:[] for k in ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']}
            parcel_penalty_split = []
            for i, neighbor in enumerate(parcel.visited):
                if i >= 3:
                    for x in ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']:
                        parcel_penalty[x].append(neighbor.node_info[x])
                else:
                    # for x in ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']:
                    #     parcel_penalty[x].append(0)
                    parcel_penalty_split.append(0)
            l = parcel.length

            for i in range(l-3):
                avg = []
                for k in ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']:
                    avg.append(np.mean([parcel_penalty[k][i],parcel_penalty[k][i-1], parcel_penalty[k][i-2]]))
                (penalty_win_lower, penalty_win_upper) = self.calc_penalty_window_ul(avg)
                parcel_penalty_split.append(round(penalty_win_lower + penalty_win_upper,6))

            penalty_window.append(sum(parcel_penalty_split))

        return sum(penalty_window)

    def calc_penalty_local_temp(self, local_solution, initial_solution):
        print('x')

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
                    local_solution.parcel_list[id].penalty_main = penalty_parcel
                    local_solution.parcel_list[id].penalty_mineral_avg = new_parcel_avg_penalty[id]
                    local_solution.viol_main = sum(x.penalty_main for x in local_solution.parcel_list)

        return local_solution

    def calc_penalty_window_ul(self, input):
        input = np.array(input)
        violation_lower = np.divide(input, self.problem.lower_limits_window_array,
                                    out=np.ones(input.size, dtype=float) * np.finfo(
                                        np.float32).max,
                                    where=self.problem.lower_limits_window_array != 0) - 1
        penalty_lower = abs(sum([x if x < 0 else 0 for x in violation_lower]))
        violation_upper = (input / self.problem.upper_limits_window_array) - 1
        penalty_upper = sum([x if x > 0 else 0 for x in violation_upper])
        return penalty_lower, penalty_upper


    def calc_penalty_upper_lower(self,input):
        violation_lower = np.divide(input, self.problem.lower_limits_array,
                                    out=np.ones(input.size, dtype=float) * np.finfo(
                                        np.float32).max,
                                    where=self.problem.lower_limits_array != 0) - 1
        penalty_lower = abs(sum([x if x < 0 else 0 for x in violation_lower]))
        violation_upper = (input / self.problem.upper_limits_array) - 1
        penalty_upper = sum([x if x > 0 else 0 for x in violation_upper])
        return penalty_lower, penalty_upper

    def local_search(self, initial_solution):
        S = []
        # t = Timer("example", text="Time spent: {:.2f}", logger=logging.warning)
        initial_visited = initial_solution.visited
        # t.start()
        for i in range(1, len(initial_visited)):
            for j in range(1, len(initial_visited)):
                if i != j:
                    (local_solution, condition) = self.local_operator(initial_solution, i, j)  # Numba?
                    if condition is True:
                        local_solution.viol_main = initial_solution.viol_main
                        local_solution.viol_window = initial_solution.viol_window
                        initial_solution.generate_edges()
                        local_solution.generate_edges()
                        L = []
                        for i, e in enumerate(local_solution.edges):
                            if e not in initial_solution.edges:
                                L.append([i + 1, e])
                        if self.valid_local(L, local_solution, i, j):
                            local_solution = self.calc_penalty_local(local_solution, initial_solution)
                            local_solution = self.calc_cost_local(local_solution, L)
                            penalty_window = self.calc_penalty_window_local(local_solution)
                            local_solution.viol_window = penalty_window
                            S.append((i, j, local_solution))
        L = [x[2] for x in S]
        cost = [x.obj for x in L]
        penalty_main = [x.viol_main for x in L]
        penalty_window = [x.viol_window for x in L]
        rank = np.lexsort((cost, penalty_window, penalty_main))
        # t.stop()
        return L[rank[0]]






