from utils import np, Solution

class PSRH():
    # Pilgrim Step Reclaiming Heuristic algorithm
    def __repr__(self):
        return f'PSRH(utility={self.total_utility}, penalty_solution={self.total_penalty_solution}, penalty_parcel={self.total_penalty_parcel})' #(id={self.name})'
    # problem : D_nodes, D_cost, number_demand, entry_cut
    def __init__(self,problem):
        self.max_capacity = np.round(problem.number_demand * 1e5)
        self.problem = problem
        self.solution = {k:Solution() for k in np.arange(self.problem.number_demand)+1}
        self.solution_ids = []
        self.solution_path=[]
        # save rng seed
        self.setup_PSRH_sequence()
        self.active_demand = 1

    @property
    def total_utility(self):

        return np.sum([v.total_utility for k,v in self.solution.items()])

    @property
    def total_penalty_solution(self):

        return np.sum([v.total_penalty_solution for k,v in self.solution.items()])

    @property
    def total_penalty_parcel(self):

        return np.sum([v.total_penalty_parcel  for k,v in self.solution.items()])

    def __len__(self):
        return len(self.solution[self.active_demand].solution_nodes)

    def __contains__(self, node_id):
        return node_id in self.solution[self.active_demand].solution_ids

    @property
    def current(self):
        if len(self) != 0:
            return (self.solution[self.active_demand].solution_nodes[-1][0],self.solution[self.active_demand].solution_nodes[-1][1])
        else:
            return (self.solution[self.active_demand-1].solution_nodes[-1][0],self.solution[self.active_demand-1].solution_nodes[-1][1])

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
        self.solution_ids.append(end_id)
        self.solution[self.active_demand].solution_ids.append(end_id)

        self.solution[self.active_demand].solution_path.append(x)

        if len(self)==1:
            self.solution[self.active_demand].average_chemical = end_node.chemical.copy()
        else:
        # self.solution[self.active_demand].average_chemical += (end_node.chemical - self.solution[self.active_demand].average_chemical)/(len(self)+1)
            self.solution[self.active_demand].average_chemical = np.array([x[0].chemical for x in self.solution[self.active_demand].solution_nodes]).mean(axis=0)
        self.solution[self.active_demand].solution_tonnage += end_node.cut_tonnage
        # make a parcel if parcel is true

    def setup_PSRH_sequence(self):
        self.seq = []
        for k in np.arange(self.problem.number_stockpile)+1:
            for j in np.arange(10)+1:
                for i in np.arange(4)+1:
                    if j < 10:
                        number_col = '0' + str(j)
                    else:
                        number_col = j
                    self.seq.append('01-01-0{}-0{}-{}'.format(k,i,number_col))


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
        del self.seq[0]
        termination = False
        while termination == False:
            # while len(self) < 160:
            end_id = self.seq[0]

            next_node_fitness = (self.calc_fitness(self.current[0].index,end_id,self.current[1],'SN'))
            next_direction = 'SN'
            self.reclaim_cut(end_id=end_id,end_direction=next_direction,end_fitness=next_node_fitness)
            del self.seq[0]

            # repeat these steps
            self.solution[self.active_demand].solution_penalty = next_node_fitness[1]
            self.solution[self.active_demand].solution_penalty_parcel.append(next_node_fitness[2])
            if self.problem.scenario == 1:
                if len(self) == self.problem.number_stockpile*40:
                    termination = True
                    print('PSRH is done \n')
                    print('Final utility cost: {}       final penalty: {}'.format(np.sum(self.solution[self.active_demand].solution_cost),self.solution[self.active_demand].solution_penalty))
                    # x
            elif self.problem.scenario == 2:
                if self.solution[self.active_demand].solution_tonnage > max_tonnage_demand:
                    # first demand is done
                    self.active_demand += 1
                    # next capacity is : max_tonnage_demand += 1e5
                    print('first demand is done')
                    if self.active_demand > self.problem.number_demand:
                        termination = True
                        print('PSRH is done \n')
                        final_utility = np.sum([np.sum(self.solution[k].solution_cost) for k in np.arange(self.problem.number_demand)+1])
                        final_penalty = np.sum([np.sum(self.solution[k].solution_penalty) for k in np.arange(self.problem.number_demand)+1])
                        final_penalty_parcel = np.sum([np.sum(self.solution[k].solution_penalty_parcel) for k in np.arange(self.problem.number_demand)+1])
                        print('Final utility cost: {}       final penalty: {}         final penalty parcel: {}'.format(final_utility, final_penalty, final_penalty_parcel))
                        for k in np.arange(self.problem.number_demand)+1:
                            print(self.solution[k])

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
        if self.problem.parcel:
            penalty_parcel = self.calc_penalty_parcel(end_id)
            return (utility,penalty_solution,penalty_parcel)
        else:
            return (utility,penalty_solution)



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