from core_aco.algorithm import np, pd
from core_aco.node import Solution
from itertools import islice
from operator import attrgetter
import copy
import logging



class Ant():
    "An ant constructs a solution using provided parameters and keeps that solution"
    def __repr__(self):
        return f'Ant(obj={self.solution.obj}, violation={self.solution.viol})'
        # return f'Ant(obj={self.objective}, violation={self.violation}, fitness={self.fitness})'

    # def __le__(self, other):
    #     "quantative comparison of ants is based on its obtained objectives"
    #     return (self.violation == other.violation) and (self.objective <= other.objective)

    def __lt__(self, other):
        # lexicographic
        # return (self.solution.viol < other.solution.viol) or ((self.solution.viol == other.solution.viol) and (self.solution.obj < other.solution.obj))
        return self.solution < other.solution

    def __contains__(self, edge):
        return edge in self.edges #or node == self.pos_current

    def __len__(self):
        return len(self.solution)

    def __init__(self, problem, start, df_ph):
        # problem.clean()
        self.problem = problem
        self.total_capacity = np.round(self.problem.demand*1e5)
        self.termination = False
        # self.active_nodes = self.problem.stockpile_entry.copy()
        self.move_count = 1
        self.start = start
        # self.df_csv = pd.DataFrame(
        #     columns=['Node', 'Cost', 'Direction', 'Constraint violation',
        #              'Tonnage reclaimed', ])  # 'Tonnage available'])
        self.df_ph = df_ph
        self.edges = []
        self.solution = Solution(None)
        self.reclaimed_cuts = {}
        self.reclaimed_cuts_keys = []
        self.setup_accessible_nodes()
        self.df_nodes = problem.df_nodes
        self.count_local = 0
        self.pos_reclaimer = None

    def reclaim_initial(self,initial_node):
        node = self.problem.df_nodes_dic[initial_node]
        self.solution.obj = node.cost_reclaim / node.cut_tonnage
        self.solution.tonnage_so_far = node.cut_tonnage
        direction = 'SN'
        del self.accessible_nodes[direction][initial_node]
        node_info = node.node_info
        for key in self.solution.visited_info:
            self.solution.visited_info[key].append(node_info[key])
        append_node = (node, direction, self.solution.obj)
        self.solution.visited.append(append_node)
        self.reclaimed_cuts[node.name] = append_node
        self.reclaimed_cuts_keys.append(node.name)
        self.solution.make_parcel(node)

    def construct_solution_case_1(self):
        self.total_capacity = np.round(self.problem.demand * self.problem.total_tonnage, 4)
        self.reclaim_initial('01-01-01-01-01')
        self.termination = False
        while self.termination == False:
            neighbor_lst = []
            for direction in self.problem.directions:
                neighbor_lst.extend(self.find_neighborhood(direction))
            for direction in self.problem.directions:
                neighbor_lst.extend(self.accessible_nodes[direction].values())
            fitness_candidates = self.evaluate_node(neighbor_lst)
            ph_info = self.evlauate_ph(neighbor_lst)

            next_idx = self.greedy_selection_constrained(fitness_candidates, ph_info)
            next_node = neighbor_lst[next_idx][0]
            next_node_direction = neighbor_lst[next_idx][1]
            self.solution.obj += fitness_candidates[next_idx][0]  # cost
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
            del (neighbor_lst[next_idx])
            self.update_accessible_nodes(neighbor_lst, next_node.name)  # accessible nodes in the stockyard for next iteration
            if np.round(self.solution.tonnage_so_far,4) >= np.round(self.total_capacity,4):
                self.termination = True
                self.solution.parcel_list[-1].end = len(self.solution) - 1
                L = [x[0] for x in self.solution.visited]
                self.solution.parcel_list[-1].visited = L[self.solution.parcel_list[-1].start:
                                                          self.solution.parcel_list[-1].end + 1]

    def construct_solution_case_2(self):
        self.reclaim_initial('01-01-01-01-01')
        for demand in range(self.problem.demand):
            self.termination = False
            while self.termination == False:
                neighbor_lst = []
                for direction in self.problem.directions:
                    neighbor_lst.extend(self.find_neighborhood(direction))
                for direction in self.problem.directions:
                    neighbor_lst.extend(self.accessible_nodes[direction].values())
                fitness_candidates = self.evaluate_node(neighbor_lst)
                ph_info = self.evlauate_ph(neighbor_lst)

                next_idx = self.greedy_selection_constrained(fitness_candidates, ph_info)
                next_node = neighbor_lst[next_idx][0]
                next_node_direction = neighbor_lst[next_idx][1]
                self.solution.obj += fitness_candidates[next_idx][0]  # cost
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
                del (neighbor_lst[next_idx])
                self.update_accessible_nodes(neighbor_lst, next_node.name)  # accessible nodes in the stockyard for next iteration
                if np.round(self.solution.tonnage_so_far,4) >= 1e5*(demand+1):
                    self.termination = True
                    self.solution.parcel_list[-1].end = len(self.solution) - 1
                    L = [x[0] for x in self.solution.visited]
                    self.solution.parcel_list[-1].visited = L[self.solution.parcel_list[-1].start:
                                                              self.solution.parcel_list[-1].end + 1]
                    # print(len(self.solution))
                    if demand + 2 <= self.problem.demand:
                        self.solution.make_parcel(None)

    def calc_penalty(self, neighbor_info):
        penalty_window = 0
        neighbor_info_properties = np.array([neighbor_info[x] for x in ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']])
        # add_penalty:
        n = self.solution.parcel_list[-1].length
        if n > 0:
            avg = self.solution.parcel_list[-1].penalty_mineral_avg
            average_limits = avg + ((neighbor_info_properties - avg) / (n + 1))
        else:
            average_limits = neighbor_info_properties
            self.solution.parcel_list[-1].start = len(self.solution)
            # if len(self.solution) == 1:
            #     self.solution.parcel_list[-1].start = 1
            # else:
            #     +1
        (penalty_lower, penalty_upper) = self.calc_penalty_upper_lower(average_limits)
        penalty_neighbor = penalty_lower + penalty_upper + penalty_window
        # if len(self.solution)==88:
        #     print('CHECK')
        return (penalty_neighbor, average_limits)

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

    def rwheel(self,fitness,ph_info):
        s = [(1 / i) ** self.problem.beta for i in fitness] # obj
        p = [i**self.problem.alpha for i in ph_info]
        x = np.multiply(s,p)
        ss = [i/sum(x) for i in x]
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
        lst_cost = [i[0][0] for i in cluster if i[0][0] > 0]
        for solution in cluster:
            obj_cost = self.mynormal(solution[0][0], min(lst_cost), max(lst_cost), 1)
            if len(lst_penalty_main) > 0:
                obj_penalty_main = self.mynormal(solution[0][1], min(lst_penalty_main), max(lst_penalty_main), 100)
            else:
                obj_penalty_main = 0

            obj = obj_cost + obj_penalty_main
            X.append(obj)
        return X

    def greedy_selection_constrained(self, fitness, ph_info):
        'FIND CLUSTER'
        Feasible = []
        InFeasible = []
        for idx, i in enumerate(fitness):
            if i[1] == 0:
                Feasible.append((i, idx))
            else:
                InFeasible.append((i, idx))
        X = []
        mapping = []
        for cluster in [Feasible, InFeasible]:
            if len(cluster) > 0:
                X.extend(self.calc_normal_cluster(cluster))
                mapping.append([i[1] for i in cluster])

        fitness_ext = []
        fitness_ext.extend(X)

        map_ext = []
        for i in mapping:
            map_ext.extend(i)

        print(fitness_ext)
        winner_idx = map_ext[self.rwheel(fitness_ext, ph_info)]
        return winner_idx

    def make_edges(self):
        self.edges = []
        temp = self.solution.visited
        while len(self.edges) < len(self)-1:
            Job_Step_2 = temp[-1][0].name
            Job_Step_1 = temp[-2][0].name
            Job_Direction_2 = temp[-1][1]
            Job_Direction_1 = temp[-2][1]
            key = (Job_Step_1, Job_Step_2, Job_Direction_1, Job_Direction_2)
            self.edges.append(key)
            temp = temp[:-1]
        self.edges.reverse()

    def find_neighborhood(self, direction):
        # find neighbors in the current stockpile & add the accessible nodes to the end
        lst = []
        (pos_reclaimer, direction_reclaimer) = self.pos_current
        neighbor_list = self.problem.df_nodes_dic[pos_reclaimer].prec_1[direction_reclaimer]
        if len(neighbor_list) > 0:
            for node in list(neighbor_list.Job_Step_2):
                if not self.reclaimed(node) and node not in self.accessible_nodes[direction].keys():
                    prec = self.problem.df_nodes_dic[node].prec[direction]
                    if len(prec) > 0:
                        if all([self.reclaimed(x) for x in list(prec.Job_Step_1)]):
                            lst.append(node)
        output = []
        for node in lst:
            # output.append((node, direction))
            output.append((self.problem.df_nodes_dic[node], direction))
        return output

    def evaluate_node(self, neighbors):
        fitness_candidates = []
        for neighbor in neighbors:
            neighbor_name = neighbor[0].name
            neighbor_direction = neighbor[1]
            # t.start()
            neighbor_info = neighbor[0].node_info
            penalty_neighbor, average_limits = self.calc_penalty(neighbor_info)
            # t.stop()
            # t.start()
            (pos_reclaimer, direction_reclaimer) = self.pos_current
            cost_move = self.problem.df_cost[(pos_reclaimer, neighbor_name,
                                                    direction_reclaimer, neighbor_direction)]['Cost']
            cost_reclaim = neighbor_info.Cost
            cost_total = cost_move + cost_reclaim
            # (penalty_neighbor, penalty_report) = self.calc_penalty(visited_temp)
            cost_neighbor = cost_total / neighbor_info.Cut_Tonnage  # + self.problem.penalty_coeff*penalty_neighbor
            cost_neighbor = round(cost_neighbor, 6)
            penalty_neighbor = round(penalty_neighbor, 6)
            # penalty_report = round(penalty_report, 6)
            fitness_candidates.append((float(cost_neighbor), penalty_neighbor, average_limits))  # , penalty_report))
            # t.stop()
        return fitness_candidates


    def evlauate_ph(self,neighbors):
        ph_info = [self.df_ph[self.pos_current[0], node[0].name, self.pos_current[1], node[1]] for node in neighbors]
        return ph_info

    def report_csv(self, cut, direction, cost, violation, tonnage):
        self.df_csv = self.df_csv.append({'Node':cut, 'Direction':direction, 'Cost':cost, 'Constraint violation':violation, 'Tonnage reclaimed': tonnage}, ignore_index=True)

    def setup_accessible_nodes(self):
        self.accessible_nodes = {k: {} for k in self.problem.directions}
        for k, v in self.problem.stockpile_entry.items():
            for node_name in v:
                value = (self.problem.df_nodes_dic[node_name], k)
                self.accessible_nodes[k].update({node_name: value})

    def update_accessible_nodes(self, neighbors, next_node_name):
        for direction in self.problem.directions:
            if next_node_name in self.accessible_nodes[direction]:
                del self.accessible_nodes[direction][next_node_name]

        for node in neighbors:
            key = node[0].name
            direction = node[1]
            if key not in self.accessible_nodes[direction]:
                self.accessible_nodes[direction].update({key: node})
            if key in self.accessible_nodes[direction] and key == next_node_name:
                del self.accessible_nodes[direction][next_node_name]

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
        # t.start()
        edges = [(x[0] - 1, x[0]) for x in L]
        for edge in edges:
            cost_neighbor = self.calc_cost_edge(edge, local_solution)
            local_solution.visited[edge[1]] = (
                local_solution.visited[edge[1]][0], local_solution.visited[edge[1]][1], cost_neighbor)
        local_solution.obj = sum([x[2] for x in local_solution.visited])
        return local_solution

    # def calc_cost_local(self, local_solution, segment):
    #     for edge in segment:
    #         cost_neighbor = self.calc_cost_edge(edge, local_solution)
    #         local_solution.visited[edge[1]] = (
    #         local_solution.visited[edge[1]][0], local_solution.visited[edge[1]][1], cost_neighbor)
    #     local_solution.obj = sum([x[2] for x in local_solution.visited])
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

    def local_search(self, initial_solution):
        S = []
        initial_visited = initial_solution.visited
        # t.start()
        for i in range(1, len(initial_visited)):
            for j in range(1, len(initial_visited)):
                if i != j:
                    try:
                        (local_solution, condition) = self.local_operator(initial_solution, i, j)
                    except:
                        print('E')
                    if condition is True:
                        local_solution.viol = initial_solution.viol
                        initial_solution.generate_edges()
                        local_solution.generate_edges()
                        L = []
                        for i, e in enumerate(local_solution.edges):
                            if e not in initial_solution.edges:
                                L.append([i + 1, e])
                        if self.valid_local(L, local_solution, i, j):
                            local_solution = self.calc_cost_local(local_solution, L)
                            local_solution = self.calc_penalty_local(local_solution, initial_solution)
                            S.append((i, j, local_solution))
        L = [x[2] for x in S]
        cost = [x.obj for x in L]
        v = [x.viol for x in L]
        rank = np.lexsort((cost, v))
        # t.stop()
        return L[rank[0]]

    def local_search_iterative(self):
        # t.start()
        self.solution.reclaimed_cuts_keys = self.reclaimed_cuts_keys
        initial_solution = self.solution
        termination = False
        while termination is False:
            local_solution = self.local_search(initial_solution)
            if local_solution < initial_solution:
                self.count_local += 1
                # print(local_solution,initial_solution)
                initial_solution = local_solution
            else:
                termination = True
        self.solution = initial_solution
        # t.stop()

    @property
    def pos_current(self):
        # return (self.solution.visited[-1][0].name, self.solution.visited[-1][1])
        index = self.reclaimed_cuts_keys[-1]
        return (index, self.reclaimed_cuts[index][1])

    def reclaimed(self,node):
        return node in self.reclaimed_cuts_keys
        # return node in [x[0].name for x in self.solution.visited]