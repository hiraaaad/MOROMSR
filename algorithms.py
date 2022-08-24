from utils import np, Solution, pickle

class GA():
    # greedy algorithm
    def __repr__(self):
        return f'Agent(utility={self.total_utility}, penalty_solution={self.total_penalty_solution}, penalty_parcel={self.total_penalty_parcel})' #(id={self.name})'
    # problem : D_nodes, D_cost, number_demand, entry_cut
    def __init__(self,problem):
        self.max_capacity = np.round(problem.number_demand * 1e5)
        self.problem = problem
        self.solution = {k:Solution() for k in np.arange(self.problem.number_demand)+1}
        self.setup_active_cuts()
        self.active_demand = 1 # demands start from 1
        self.solution_ids = []
        self.solution_path = []
        # save rng seed

    def __len__(self):
        return len(self.solution[self.active_demand].solution_nodes)


    def __contains__(self, node_id):
        return node_id in self.solution_ids


    @property
    def total_utility(self):

        return np.sum([v.total_utility for k,v in self.solution.items()])

    @property
    def total_penalty_solution(self):

        return np.sum([v.total_penalty_solution for k,v in self.solution.items()])

    @property
    def total_penalty_parcel(self):

        return np.sum([v.total_penalty_parcel  for k,v in self.solution.items()])

    def reclaim_cut(self,end_id, end_direction, end_fitness):
        """
        This function reclaims a cut in the stockyard for the agent
        :param end_id: id of node to be reclaimed
        :param end_direction: the direction of reclaiming of reclaiming the node
        :param end_fitness: calculated tuple of fitness for the end node
        :return: updates the solution and active nodes
        """
        end_node = self.problem.D_nodes[end_id]

        if len(self.solution_ids)>0:
            path = (self.current[0].index,end_id,self.current[1],end_direction)
            self.solution[self.active_demand].solution_path.append(path)
            self.solution_path.append(path)

        self.solution[self.active_demand].solution_nodes.append((end_node, end_direction))
        self.solution[self.active_demand].solution_cost.append(end_fitness[0])
        # self.solution[self.active_demand].solution_ids.append(end_id)
        self.solution_ids.append(end_id)
        self.solution[self.active_demand].solution_ids.append(end_id)

        # update chemical
        if len(self)==1:
            self.solution[self.active_demand].average_chemical = end_node.chemical.copy()
        else:
            # self.solution[self.active_demand].average_chemical += (end_node.chemical - self.solution[self.active_demand].average_chemical)/(len(self)+1)
            self.solution[self.active_demand].average_chemical = np.array([x[0].chemical for x in self.solution[self.active_demand].solution_nodes]).mean(axis=0)
        self.solution[self.active_demand].solution_tonnage += end_node.cut_tonnage

        # update active nodes and remove the cut from the active nodes
        self.active_cuts[end_node.node_stockpile][end_direction].remove(end_id)


    @property
    def current(self):
        if len(self) != 0:
            return (self.solution[self.active_demand].solution_nodes[-1][0],self.solution[self.active_demand].solution_nodes[-1][1])
        else:
            return (self.solution[self.active_demand-1].solution_nodes[-1][0],self.solution[self.active_demand-1].solution_nodes[-1][1])

    def setup_active_cuts(self):
        """
        this function sets the entry points for the stockpiles wrt the direction
        :param number_stockpile:
        :return: setups a dictionary of active nodes in a stockyard which are the entry points at beginning
        """
        self.active_cuts = {k: dict(zip(['SN','NS'],[['01-01-0{}-01-01'.format(k)],['01-01-0{}-01-10'.format(k)]]))  for k in np.arange(self.problem.number_stockpile)+1}

    def evolve(self):
        """
        this function runs the evolution part of the algorithm to obtain an ultimate solution
        :return:
        """
        max_tonnage_demand = 1e5
        entry_cut_id = '01-01-01-01-01'
        entry_cut_direction = 'SN'
        # update chemical
        entry_cut = self.problem.D_nodes[entry_cut_id]
        self.solution[self.active_demand].average_chemical = entry_cut.chemical.copy()
        end_fitness = self.calc_fitness(start_id=None,end_id=entry_cut_id,start_direction=None,end_direction='SN') # initial reclaim
        self.reclaim_cut(end_id=entry_cut_id, end_direction=entry_cut_direction, end_fitness=end_fitness)
        termination = False
        while termination == False:
        # while len(self) < 160:
            neighbors_id_SN = []
            neighbors_id_NS = []
            fitness_neighbors_SN = []
            fitness_neighbors_NS = []
            ################################################
            # neighbors_id_SN = \
            self.find_neighbors(direction_reclaimer='SN')  # a list of active neighbors in SN direction
            # neighbors_NS = self.find_neighbors(direction_reclaimer='NS') # disable
            [neighbors_id_SN.extend(v['SN']) for k,v in self.active_cuts.items()]
            # [neighbors_id_NS.extend(v['SN']) for k,v in self.active_cuts.items()] # disable
            # accumulate neighbors
            # each neighbor should be evaluated
            [fitness_neighbors_SN.append(self.calc_fitness(self.current[0].index,end_id,self.current[1],'SN')) for end_id in neighbors_id_SN]
            # [fitness_neighbors_NS.append(self.fitness(self.current[0].index,end_id,self.current[1],'NS')) for end_id in neighbors_id_NS] # disable

            neighbors_id = neighbors_id_SN + neighbors_id_NS

            if len(neighbors_id) >0:
                fitness_neighbors = fitness_neighbors_SN + fitness_neighbors_NS

                # if choose greedy:
                if self.problem.greedy_factor == 0:
                    next_node_index = self.greedy_choose(fitness_neighbors)
                else:
                    next_node_index = self.ranomised_greedy_choose(fitness_neighbors)

                if next_node_index < len(neighbors_id_SN):
                    next_direction = 'SN'
                else:
                    next_direction = 'NS'

                next_node_id = neighbors_id[next_node_index]
                next_node_fitness = fitness_neighbors[next_node_index]

                self.reclaim_cut(end_id=next_node_id,end_direction=next_direction,end_fitness=next_node_fitness)
                self.solution[self.active_demand].solution_penalty = next_node_fitness[1]
                if self.problem.parcel:
                    self.solution[self.active_demand].solution_penalty_parcel.append(next_node_fitness[2])

            # repeat these steps
            if self.problem.scenario == 1:
                if len(self) == self.problem.number_stockpile*40:
                    termination = True
                    if self.problem.greedy_factor == 0:
                        alg_str = 'GA'
                    else:
                        alg_str = 'RGA'
                    print('{} is done \n'.format(alg_str))
                    print('Final utility cost: {}       final penalty: {}'.format(np.round(np.sum(self.solution[self.active_demand].solution_cost),self.problem.precision),self.solution[self.active_demand].solution_penalty))
                    # x
            elif self.problem.scenario == 2:
                if self.solution[self.active_demand].solution_tonnage > max_tonnage_demand:
                    # first demand is done
                    self.active_demand += 1
                    # next capacity is : max_tonnage_demand += 1e5
                    print('first demand is done')
                    if self.active_demand > self.problem.number_demand:
                        termination = True
                        if self.problem.greedy_factor == 0:
                            alg_str = 'GA'
                        else:
                            alg_str = 'RGA'
                        print('{} is done \n'.format(alg_str))
                        final_utility = np.round(np.sum([np.sum(self.solution[k].solution_cost) for k in np.arange(self.problem.number_demand)+1]),self.problem.precision)
                        final_penalty = np.round(np.sum([np.sum(self.solution[k].solution_penalty) for k in np.arange(self.problem.number_demand)+1]),self.problem.precision)
                        final_penalty_parcel = np.sum([np.sum(self.solution[k].solution_penalty_parcel) for k in np.arange(self.problem.number_demand)+1])
                        print('Final utility cost: {}       final penalty: {}         final penalty parcel: {}'.format(final_utility, final_penalty, final_penalty_parcel))
                        for k in np.arange(self.problem.number_demand)+1:
                            print(self.solution[k])

    def find_neighbors(self,direction_reclaimer):
        """
        :type direction_reclaimer: str
        :param direction_reclaimer: to reclaim in 'SN' or 'NS' direction for next cut
        :return: s list of cut_id that are active neighbors wrt the direction for the current stockpile of the the current position of agent
        """
        neighbor_list = self.current[0].prec_1[direction_reclaimer]
        # lst = {k:[] for k in np.arange(self.problem.number_stockpile)+1}
        lst = []
        if len(neighbor_list) > 0:
            for neighbor_id in neighbor_list:
                neighbor = self.problem.D_nodes[neighbor_id]
                if neighbor_id not in self and neighbor_id not in self.active_cuts[neighbor.node_stockpile][direction_reclaimer]:
                    prec = neighbor.prec[direction_reclaimer]
                    if len(prec) > 0:
                        if all(node in self for node in prec):
                            # lst[neighbor.node_stockpile].append(neighbor_id) # it is a valid neighbor
                            lst.append(neighbor_id)
                    # self.accessible_nodes[direction].keys():
            # for k in np.arange(self.problem.number_stockpile)+1:
            #     if k != self.current[0].node_stockpile:
            #         lst[k].extend(self.active_cuts[k][direction_reclaimer])

        # update active nodes and add the found neighbors
        # self.active_cuts[end_node.node_stockpile][end_direction].remove(end_id)
        # for k,v in lst.items():
        #     for node in v:
        #         if node not in self.active_cuts[k][direction_reclaimer]:
        #             self.active_cuts[k][direction_reclaimer].append(node)
        [self.active_cuts[neighbor.node_stockpile][direction_reclaimer].append(node) for node in lst]

        # return lst

    def choose_next_randomly(self,lst):
        """
        this function is just a test : to choose the successor node randomly
        :param lst:
        :return: one index from lst randomly
        """
        choose = np.random.choice(range(len(lst)),1)[0]
        return choose

    def calc_penalty_solution(self,node_id):
        """
        this function calculates the penalty of solution for a node in question :: first case and second case
        :param node_id:
        :return: total violation, calculated average chemical
        """
        node = self.problem.D_nodes[node_id]
        temp_average_chemical = self.solution[self.active_demand].average_chemical + (node.chemical - self.solution[self.active_demand].average_chemical)/(len(self)+1)
        return self.calc_violation(temp_average_chemical,1) # total solution violation

    def calc_penalty_parcel(self,node_id):
        if len(self)>=4:
            node_current = self.problem.D_nodes[node_id] # current
            node_1 = self.solution[self.active_demand].solution_nodes[-1][0] # ultimate node
            node_2 = self.solution[self.active_demand].solution_nodes[-2][0] #penultimate node
            parcel_average_chemical = np.array([node.chemical for node in [node_current,node_1,node_2]]).mean(axis=0)
            return self.calc_violation(parcel_average_chemical,2) # parcel violation
        else:
            return np.float(0)

    def calc_violation(self,array_average,case):
        if case == 1:
            min_limit = self.problem.limits_lower_solution
            max_limit = self.problem.limits_upper_solution
        else:
            min_limit = self.problem.limits_lower_window
            max_limit = self.problem.limits_upper_window

        # we see only Fe has a lower bound therefore the lower bound violation is calculated only considering Fe
        # :: ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']
        # we calculate the violation of bounds by bracket operator
        viol_lower = np.abs(min((array_average[1]-min_limit[1])/min_limit[1],0))
        viol_upper = (max_limit-array_average)/max_limit
        viol_upper[viol_upper > 0] = 0
        viol_upper = np.sum(abs(viol_upper))
        viol = np.round(viol_lower + viol_upper, self.problem.precision)
        return viol

    def calc_fitness(self,start_id,end_id,start_direction,end_direction):
        """
        this function calculates the tuple of fitness for a node
        :param start_id: index id of the current position of reclaimer
        :param end_id: id of the node to be evaluated
        :param start_direction: current direction of the reclaimer
        :param end_direction: direction of reclaiming
        :return: the tuple of fitness for a node
        """
        # to evlauate fitness for each node in the neighborhood
        end_node = self.problem.D_nodes[end_id]

        if start_id is None and start_direction is None:
        # initial reclaim
            cost_moving = 0
        else:
            cost_moving = self.problem.D_cost[start_id,end_id,start_direction,end_direction]

        cost_reclaim = end_node.cost_reclaim

        cost = cost_reclaim + cost_moving
        cost = np.round(cost,self.problem.precision)
        end_cut_tonnage = np.round(end_node.cut_tonnage,self.problem.precision)
        utility = cost/end_cut_tonnage
        utility = np.round(utility,self.problem.precision)
        penalty_solution = self.calc_penalty_solution(end_id)
        penalty_solution = np.round(penalty_solution,self.problem.precision)
        if self.problem.parcel:
            penalty_parcel = self.calc_penalty_parcel(end_id)
            return (utility,penalty_solution,penalty_parcel)
        else:
            return (utility,penalty_solution)

    def greedy_choose(self,fitness_neighbors):
        """
        this function perform the greedy choosing for the next node from the active nodes in the neighborhood
        :param fitness_neighbors: fitness of active neighbors
        :return: index of chosen node from the neighboorhood
        """
        # to choose the next node in the most greedy form
        penalty_solution = [x[1] for x in fitness_neighbors] # First column
        cost = [x[0] for x in fitness_neighbors] # Second column
        # penalty_parcel = % disable
        if self.problem.parcel:
            penalty_parcel = [x[2] for x in fitness_neighbors]
            sorted_neighbors = np.lexsort((cost,penalty_parcel,penalty_solution)) # Sort by penalty, then by cost
        else:
            sorted_neighbors = np.lexsort((cost,penalty_solution)) # Sort by penalty, then by cost
        return sorted_neighbors[0]

    def ranomised_greedy_choose(self,fitness_neighbors):
        """
        this function performs a randomised greedy selection on the active nodes in the neighboorhood
        :param fitness_neighbors: fitness of active neighbors
        :return: index of chosen node from the neighboorhood
        """
        # we choose the next node acoording to \lambda greediness of the algorithm
        if len(fitness_neighbors) > 1:
            penalty_solution = np.array([x[1] for x in fitness_neighbors])# First column
            cost = np.array([x[0] for x in fitness_neighbors]) # Second column
            if self.problem.parcel:
                penalty_parcel = np.array([x[2] for x in fitness_neighbors])
            else:
                penalty_parcel = np.zeros(len(fitness_neighbors))

            norm_cost = (cost-min(cost))/(max(cost)-min(cost))
            norm_penalty_solution = np.divide((penalty_solution-min(penalty_solution)),(max(penalty_solution)-min(penalty_solution)),out=np.zeros_like(penalty_solution), where=(max(penalty_solution)-min(penalty_solution))!=0)
            norm_penalty_parcel = np.divide((penalty_parcel-min(penalty_parcel)),(max(penalty_parcel)-min(penalty_parcel)),out=np.zeros_like(penalty_parcel), where=(max(penalty_parcel)-min(penalty_parcel))!=0)
            # determine sets
            S1 = np.where(np.logical_and(penalty_solution==0,penalty_parcel ==0)) # feasible solutions
            S2 = np.where(np.logical_and(penalty_solution==0,penalty_parcel !=0)) # infeasible only wrt penalty_parcel
            S3 = np.setdiff1d(np.array(range(len(fitness_neighbors))),np.concatenate((S1[0],S2[0]),axis=0)) # otherwise
            norm_fitness_neighbors = norm_cost + norm_penalty_solution + norm_penalty_parcel
            norm_fitness_neighbors[S1] += 1
            norm_fitness_neighbors[S2] += 11
            norm_fitness_neighbors[S3] += 111
            ## rwheel
            return self.rwheel(norm_fitness_neighbors)
        else:
            return 0

    def rwheel(self,lst):
        """
        this function performs a roulette-wheel to do a fitness proportionate selection
        :param lst: the normalised fitness
        :return: the chosen index from the list
        """
        # this part of code has been inspired from https://stackoverflow.com/a/52243810/5582927
        s = (1/lst)**self.problem.greedy_factor
        selection_probs = s/sum(s)
        # csum = np.cumsum(selection_probs) :: it should be 1.0
        # if  is np.nan:
        return np.random.choice(len(lst), p=selection_probs)