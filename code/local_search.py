from utils import np, Solution, pickle
class LocalSearch():
    """
    this class takes a dictionary of solutions, total obtained path and current fitness of total solution and performs local search
    """
    def __repr__(self):
        return f'LocalSearch Agent'  #(utility={self.total_utility}, penalty_solution={self.total_penalty_solution}, penalty_parcel={self.total_penalty_parcel})' #(id={self.name})'
    def __init__(self, problem, operator: int, D_solution: dict, solution_ids : list):
        self.operator = operator
        self.D_solution = D_solution # dictionary of solutions
        self.solution_ids = solution_ids
        self.problem = problem

    def swap(self,lst,a,b):
        temp = lst[a]
        lst[a] = lst[b]
        lst[b] = temp

    def insert(self,lst,a,b): # move the second omponent ahead of the first componet
        temp = lst[b]
        for i in range(0,b-a-1):
            lst[b-i] = lst[b-(i+1)]
        lst[a+1] = temp

    def inverse(self,lst,a,b):
        for i in range(0,int(np.ceil((b-a)/2))):
            temp = lst[a+i]
        lst[a+i] = lst[b-i]
        lst[b-i] = temp

    def generate_span(self,operator,lst,a,b):
        if operator ==0 or operator == 2:
            return lst[a:b+1]
        elif operator == 1:
            return lst[a+1:b+1]


    def check_validity_swap(self,lst,a,b):
        # Does the component in lst[a] has any precedence constraints now ahead for prescribed inteval
        prec_a = self.problem.D_nodes[lst[a]].prec['SN']
        # does the component in lst[b] is a precedence itself for any item head
        prec_1_b = self.problem.D_nodes[lst[b]].prec_1['SN']
        span = self.generate_span(operator=0, lst=lst, a=a, b=b)
        if any(item in span for item in prec_a):
           return False
            # reject it + break
        if any(item in lst[a:b+1] for item in prec_1_b):
            return False
        return True

    def check_validity_insert(self,lst,a,b):
        # Does the component in lst[b] has any precedence in the span :: reject
        prec_b =  self.problem.D_nodes[lst[a+1]].prec['SN']
        span = self.generate_span(operator=1, lst=lst, a=a, b=b)
        if any(item in span for item in prec_b):
            return False
        return True

    def check_validity_inverse(self,lst,a,b):
        span = self.generate_span(operator=2, lst=lst, a=a, b=b)
        for item in span:
            prec_item = self.problem.D_nodes[item].prec['SN']
            if any(x in span for x in prec_item):
                return False
        return True

    def loop_swap(self):
        loop_result = []
        for a in range(1,len(self.solution_ids)):
            for b in range(1,len(self.solution_ids)):
                if a < b:
                    lst = self.solution_ids.copy()
                    self.swap(lst,a,b)
                    if self.check_validity_swap(lst,a,b):
                        recalc_solution = self.recalculate_solution(lst=lst, a=a, b=b)
                        loop_result.append((lst,recalc_solution))
        return loop_result

    def loop_insert(self):
        loop_result = []
        for a in range(1,len(self.solution_ids)):
            for b in range(1,len(self.solution_ids)):
                if a != b:
                    lst = self.solution_ids.copy()
                    self.insert(lst,a,b)
                    if self.check_validity_insert(lst,a,b) and lst != self.solution_ids:
                        recalc_solution = self.recalculate_solution(lst=lst, a=a, b=b)
                        loop_result.append((lst,recalc_solution))
        return loop_result

    def loop_inverse(self):
        loop_result = []
        for a in range(1,len(self.solution_ids)):
            for b in range(1,len(self.solution_ids)):
                if a < b:
                    lst = self.solution_ids.copy()
                    self.inverse(lst,a,b)
                    if self.check_validity_inverse(lst,a,b):
                        recalc_solution = self.recalculate_solution(lst=lst, a=a, b=b)
                        loop_result.append((lst,recalc_solution))
        return loop_result


    def run_iterative(self):
        termination = False
        while termination is False:
            # perform the local search for current initial solution
            self.operator = 0 # 0: swap, 1: insert, 2: inverse
            if self.operator == 0:
                loop_result = self.loop_swap()
            elif self.operator == 1:
                self.loop_insert()
            elif self.operator == 2:
                self.loop_inverse()

            S = []
            for item in loop_result:
                s = np.array([v.total_fitness() for k,v in item[1].items()])
                S.append(sum(s))

            cost = [x[0] for x in S]
            penalty_1 = [x[1] for x in S]
            penalty_2 = [x[2] for x in S]
            best_local_index = np.lexsort((cost,penalty_2,penalty_1))[0]

            ## compare best
            best_local_fitness = S[best_local_index]
            best_fitness = sum(np.array([v.total_fitness() for k,v in self.D_solution.items()]))

            cost_comparative = (best_fitness[0],best_local_fitness[0])
            penalty_solution_comparative = (best_fitness[1],best_local_fitness[1])
            penalty_parcel_comparative = (best_fitness[2],best_local_fitness[2])
            winner = np.lexsort((cost_comparative,penalty_parcel_comparative,penalty_solution_comparative))[0]

            if winner == 0:  # replace the previous best-found solution if current best local is better than previous best found solution
                termination = True
                # genenrate its path
                path = self.generate_path(self.solution_ids)
                return (path,best_fitness)
            elif winner == 1:
                self.D_solution = loop_result[best_local_index][1]
                self.solution_ids = loop_result[best_local_index][0]

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

    def generate_path(self,lst):
        Path = []
        for i,e in enumerate(lst):
            if i > 0:
                Path.append((lst[i-1],lst[i],'SN','SN'))
        return Path

    def recalculate_solution(self,lst,a,b):
        """
        This function recalculates the solution dictionary for the purtubed solution
        :param lst:
        :param a:
        :param b:
        :return:
        """
        # non-efficient way:
        Path = self.generate_path(lst) # storing path keys

        # reclaim the path
        active_demand = 1
        cost = [self.problem.D_nodes[self.solution_ids[0]].cost_reclaim]
        cut_tonnage = [self.problem.D_nodes[self.solution_ids[0]].cut_tonnage]

        recalc_solution = {k:Solution() for k in np.arange(self.problem.number_demand)+1}
        index = 0
        # initial reclaim
        entry_node = self.problem.D_nodes[self.solution_ids[0]]
        recalc_solution[1].solution_tonnage += entry_node.cut_tonnage
        recalc_solution[1].solution_nodes.append((entry_node,'SN'))
        recalc_solution[1].solution_ids.append(entry_node.index)
        # recalc_solution[1].solution_path.append(path)
        recalc_solution[1].solution_cost.append(entry_node.cost_reclaim/entry_node.cut_tonnage)

        path_index = 0
        temp_chemical = []
        while path_index < len(Path):
            path = Path[path_index]
            end_id = path[1]
            end_node = self.problem.D_nodes[end_id]
            try:
                recalc_solution[active_demand].solution_tonnage += end_node.cut_tonnage
            except:
                active_demand -= 1
                recalc_solution[active_demand].solution_tonnage += end_node.cut_tonnage
            recalc_solution[active_demand].solution_nodes.append((end_node,'SN'))
            recalc_solution[active_demand].solution_ids.append(end_id)
            recalc_solution[active_demand].solution_path.append(path)
            recalc_solution[active_demand].solution_cost.append((end_node.cost_reclaim + self.problem.D_cost[path])/end_node.cut_tonnage)
            temp_chemical.append(end_node.chemical)
            if self.problem.parcel:
                if len(temp_chemical)>4:
                    recalc_solution[active_demand].solution_penalty_parcel.append(self.calc_violation(np.array([temp_chemical[-1],temp_chemical[-2],temp_chemical[-3]]).mean(axis=0),2))

            path_index += 1
            if recalc_solution[active_demand].solution_tonnage > 1e5:
                temp_chemical=np.array(temp_chemical)
                temp_chemical.mean(axis=0)
                recalc_solution[active_demand].solution_penalty = self.calc_violation(temp_chemical.mean(axis=0),1)
                active_demand += 1
                temp_chemical = []

        return recalc_solution

        # efficient way in development:
        # csum = np.cumsum([len(v) for k,v in self.D_solution.items()]) #cumsum length solutions in a reclaiming schedule
        # # find extreme points:
        # extreme = [x-1 for x in csum]
        # extreme = csum.copy()
        # index_change = [np.where(x<=csum)[0][0] for x in [a,b]]
        # if index_change.count(index_change[0]) == 2: # the change only affect a single solution]
        #     # if all indices are not in the extreme points (start and end of a solution)
        #     if a>
        #     # span = self.generate_span(operator=self.operator, lst=lst, a=a, b=b)
        #     # calculate fitness in a non efficient way:
        #     # recalc cost:
        #     # the first element in the cost is the reclamation utlity of the entry point :: non-changable
        #
        #     # penalty solution does not change because the total average of chemical material does not change
        #
        # else:
        #     pass