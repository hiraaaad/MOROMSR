from core.utils import np, nx, pd
import pickle

class Problem:

    def __init__(self,
                 number_machine : int,
                 number_row : int,
                 number_stockpile : int,
                 number_bench : int,
                 number_cut : int,
                 demand : float,
                 alg_type : int,
                 greedy_alpha : int,
                 direction : int,
                 phoenix : int,
                 ls: int,
                 case: int,
                 ):
        if direction == 0:
            self.directions = ['SN']
        elif direction == 1:
            self.directions = ['NS']
        else:
            self.directions = ['SN', 'NS']

        if len(self.directions) > 1:
            direction_str = 'Bi'
        else:
            if self.directions == ['SN']:
                direction_str = 'SN'

            elif self.directions == ['NS']:
                direction_str = 'NS'

        self.df_prec = pd.read_csv('eka_testproblem/ready_problems/Precedence_{}_{}_{}_{}_{}.csv'
                                   .format(number_row, number_stockpile, number_bench, number_cut, direction_str), index_col=0)

        self.df_cost = pd.read_csv('eka_testproblem/ready_problems/Cost_{}_{}_{}_{}_{}.csv'
                                   .format(number_row, number_stockpile, number_bench, number_cut, direction_str), index_col=0)

        self.df_nodes = pd.read_csv('eka_testproblem/ready_problems/Nodes_{}_{}_{}_{}.csv'
                                    .format(number_row, number_stockpile, number_bench, number_cut), index_col=0)

        # if self.directions == ['SN']:
        #     self.G = nx.read_graphml('eka_testproblem/ready_problems/graph_stockyard_1_4_4_10_SN.graphml')  # only one machine SN
        # elif self.directions == ['NS']:
        #     self.G = nx.read_graphml(
        #         'eka_testproblem/ready_problems/graph_stockyard_1_4_4_10_NS.graphml')  # only one machine NS
        # else:
        #     self.G = nx.read_graphml(
        #         'eka_testproblem/ready_problems/graph_stockyard_1_4_4_10_Bi.graphml')  # only one machine SN

        # read pickle
        str_dic = 'eka_testproblem/ready_problems/Dictionary_Nodes_GA_{}_{}_{}_{}_{}.pickle'.format(number_row, number_stockpile,
                                                                                         number_bench, number_cut, direction_str)
        with open (str_dic,'rb') as dic_file:
            self.df_nodes_dic = pickle.load(dic_file)
        # self.df_cost = self.df_cost.set_index(["Job_Step_1", "Job_Step_2", "Job_1", "Job_2"])
        # df_limits = pd.read_csv('eka_testproblem/SyntheticData1/Limits' + '.csv').round(4)
        # self.SN = nx.read_graphml('eka_testproblem/graphml/EKA_stockyard_M1_SN.graphml') #only one machine SN
        # self.NS = nx.read_graphml('eka_testproblem/graphml/EKA_stockyard_M1_NS.graphml') #only one machine NS
        self.limits_upper = {'Al2O3': 2.3886, 'Fe': 100, 'Mn': 0.2191, 'P': 0.106, 'S': 0.0311, 'SiO2': 4.0246}
        self.limits_lower = {'Al2O3': 0, 'Fe': 61.1650, 'Mn': 0, 'P': 0, 'S': 0, 'SiO2': 0}
        self.limits_window_upper = {'Al2O3': 2.4677, 'Fe': 100, 'Mn': 0.2686, 'P': 0.109, 'S': 0.03615, 'SiO2': 4.1777}
        self.limits_window_lower = {'Al2O3': 0, 'Fe': 60.9636, 'Mn': 0, 'P': 0, 'S': 0, 'SiO2': 0}
        self.lower_limits_window_array = np.array(list(self.limits_window_lower.values()))
        self.upper_limits_window_array = np.array(list(self.limits_window_upper.values()))
        self.lower_limits_array = np.array(list(self.limits_lower.values()))
        self.upper_limits_array = np.array(list(self.limits_upper.values()))
        # self.stockpile_entry = ['01-01-01-01-01', '01-01-02-01-01', '01-01-03-01-01', '01-01-04-01-01']
        # self.stockpile_entry = ['01-01-01-01-01', '01-01-02-01-01', '01-01-03-01-01', '01-01-04-01-01']
        self.number_machine = number_machine
        self.number_row = number_row
        self.number_stockpile = number_stockpile
        self.number_bench = number_bench
        self.number_cut = number_cut

        self.stockpile_entry = {k: {} for k in self.directions}
        # self.penalty_coeff = 10**4
        # self.penalty_coeff = 1
        # self.constraint_type = constraint_type
        if greedy_alpha == 0:
            self.greedy_type = 'GA'
        elif greedy_alpha >= 1:
            self.greedy_type = 'RGA'
        self.greedy_alpha = greedy_alpha

        self.case = case
        if self.case == 2:
            self.demand = int(demand)
        else:
            self.demand = float(demand)

        self.setup_df_nodes()
        self.total_tonnage = self.df_nodes.Cut_Tonnage.sum()
        # self.penalty_step = np.ceil(np.ceil(self.df_nodes.Cost.max())/np.floor(self.df_nodes.Cut_Tonnage.min()))
        # self.accessible_nodes =
        # self.log_str = time.strftime("%Y%m%d%H%M%S") + '.log'
        self.log_str = None
        self.phoenix = bool(phoenix)
        self.ls = int(ls)
        # self.algorithm_type = algorithm_type
    def __repr__(self):
        return f'Problem_Greedy(number_machine={self.number_machine}, number_row={self.number_row}, number_stockpile={self.number_stockpile}, ' \
            f'number_bench={self.number_bench}, number_cut={self.number_cut}, demand={self.demand}, alpha={self.greedy_alpha}, direction={self.directions})'

    # def setup_accessible_nodes(self):
    #     # self.accessible_nodes = pd.DataFrame(columns=self.columns)
    #     self.accessible_nodes = pd.DataFrame(columns=['Direction'])
    #     self.accessible_nodes.index.name = 'Cut_ID'
    #     for stockpile_idx in range(self.number_stockpile):
    #         for i, direction in enumerate(self.directions):
    #             node = self.stockpile_entry[stockpile_idx][i]
    #             # node_info = self.identify_node(node)
    #             # node_info['Direction'] = direction
    #             self.accessible_nodes.loc[node]=[direction]
    #
    # def update_accessible_nodes(self,neighbors,next_node_info):
    #     # neighbors = neighbors.drop(next_node_info.name)
    #     if next_node_info.name in self.accessible_nodes.index:
    #         self.accessible_nodes = self.accessible_nodes.drop(next_node_info.name)
    #     for node in neighbors:
    #         if node[0] not in self.accessible_nodes.index:
    #             self.accessible_nodes.loc[node[0]]=node[1]
    #         else:
    #             if node[1] != self.accessible_nodes.loc[node[0]]['Direction']:
    #                 self.accessible_nodes.loc[node[0]] = node[1]



    def setup_df_nodes(self):
        if 'SN' in self.stockpile_entry:
            self.stockpile_entry['SN']=['01-01-0{}-01-01'.format(x) for x in range(1, self.number_stockpile+1)]
        if 'NS' in self.stockpile_entry:
            self.stockpile_entry['NS']=['01-01-0{}-01-{}'.format(x,self.string_cut(self.number_cut)) for x in range(1, self.number_stockpile + 1)]
    #
    #
    #     for key in
    #         for stockpile in
    #
    #             '01-01-0{}-01-01'.format(number_sockpile)
    #
    #     for key in self.stockpile_entry:
    #         if len(self.directions)>1:
    #             self.stockpile_entry[key]=['01-01-0'+str(key+1)+'-01-01', '01-01-0'+str(key+1)+'-01-'+self.string_cut(self.number_cut)]
    #         else:
    #             if self.directions[0]=='SN':
    #                 self.stockpile_entry[key] = ['01-01-0' + str(key + 1) + '-01-01']
    #             else:
    #                 self.stockpile_entry[key] = ['01-01-0'+str(key+1)+'-01-'+self.string_cut(self.number_cut)]
    # #
    #     # print('df_prec setup was successful')
    #     #
    #     self.df_cost = self.df_cost[self.df_cost.Job_Step_1.str.startswith('01-01')
    #                                 & self.df_cost.Job_Step_2.str.startswith('01-01')]
    #     self.df_cost = self.df_cost[(self.df_cost.Job_Step_1.str[-2:].astype(int) <= self.number_cut) &
    #                                 (self.df_cost.Job_Step_2.str[-2:].astype(int) <= self.number_cut)]
    #     self.df_cost = self.df_cost[(self.df_cost.Job_Step_1.str[-5:-3].astype(int) <= self.number_bench) &
    #                                 (self.df_cost.Job_Step_2.str[-5:-3].astype(int) <= self.number_bench)]
    #     self.df_cost = self.df_cost[(self.df_cost.Job_Step_1.str[-8:-6].astype(int) <= self.number_stockpile) &
    #                                 (self.df_cost.Job_Step_2.str[-8:-6].astype(int) <= self.number_stockpile)]
    #     # self.df_cost = self.df_cost.reset_index(drop=True)
    #     self.df_cost = self.df_cost.set_index(["Job_Step_1", "Job_Step_2", "Job_1", "Job_2"])
        self.df_nodes = self.df_nodes.set_index(["Cut_ID"])
        self.df_nodes = self.df_nodes.sort_index()
        self.columns = list(self.df_nodes.columns)
        self.columns.append('Direction')
        self.df_cost = self.df_cost.reset_index(drop=True)
        self.df_cost = self.df_cost.set_index(["Job_Step_1", "Job_Step_2", "Job_1", "Job_2"])
        self.df_cost = self.df_cost.to_dict(orient='index')
        self.df_prec = self.df_prec.reset_index(drop=True)
        self.df_prec_1 = self.df_prec.set_index(["Job_Step_1", "Job"])
        self.df_prec = self.df_prec.set_index(["Job_Step_2", "Job"])
        self.df_prec = self.df_prec.sort_index()
        self.df_prec_1 = self.df_prec_1.sort_index()

        print('df_cost setup was successful')

    # def identify_node(self,node):
    #     # if type(node) is list:
    #     #     for i,e in enumerate(node):
    #     #         df = pd.DataFrame(columns=self.columns)
    #     #         df.loc[i]=self.df_nodes.loc[node].copy()
    #     #     return df
    #     # else:
    #     return self.df_nodes.loc[node].copy()

    def identify_node_stockpile(self,node):
        return int(node.split('-')[2]) # to make it consistent with the

    def identify_node_bench(self,node):
        return int(node.split('-')[3])

    def identify_node_cut(self,node):
        return int(node.split('-')[4])

    def string_stockpile(self,stockpile_idx):
        if stockpile_idx >= 9:
            return str(stockpile_idx+1)
        else:
            return '0'+str(stockpile_idx+1)

    def string_cut(self, cut_idx):
        if cut_idx >= 9:
            return str(cut_idx)
        else:
            return '0' + str(cut_idx)


    def clean(self):
        # self.visited_info = pd.DataFrame(columns=self.columns)
        # self.visited_info.index.name = 'Cut_ID'
        # self.visited_cuts = pd.DataFrame(columns=['Direction'])
        # self.visited_cuts.index.name = 'Cut_ID'
        self.rng_seed = int(np.random.randint([1e6]))
        # self.rng_seed = int(92413)