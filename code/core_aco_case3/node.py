from core_aco_case3.algorithm import np
class Node():
    def __repr__(self):
        return f'Cut(id={self.name})'

    # def __init__(self, name, problem):
    #     self.node_info = problem.df_nodes.loc[name]
    #     self.name = name
    #     # self.direction = direction
    #     self.cost_reclaim = self.node_info['Cost']
    #     self.cut_tonnage = self.node_info['Cut_Tonnage']

    def __init__(self,name, df_nodes, df_prec, df_prec_1):
        self.node_info = df_nodes.loc[name]
        self.name = name
        # self.direction = direction
        self.cost_reclaim = self.node_info['Cost']
        self.cut_tonnage = self.node_info['Cut_Tonnage']
        self.prec = {'SN':[],'NS':[]}
        self.prec_1 = {'SN':[],'NS':[]}
        for direction in ['SN','NS']:
            if (self.name, direction) in df_prec.index:
                self.prec[direction] = df_prec.loc[self.name, direction]
            # else:
            #     self.prec[direction] = []

            if (self.name, direction) in df_prec_1.index:
                self.prec_1[direction] = df_prec_1.loc[self.name, direction]

class Parcel():
    def __repr__(self):
        return f'(penalty_avg={self.penalty_main}, penalty_window={self.penalty_window_total})'

    def __init__(self):
        self.penalty_mineral_avg = np.array([]) # = ['Al2O3', 'CaO', 'Fe', 'MgO', 'Mn', 'P', 'S', 'SiO2', 'TiO2']
        self.penalty_window = []
        self.penalty_main = np.inf
        # self.tonnage = 0
        self.length = 0
        self.start = 0
        self.end = 0

    @property
    def penalty_window_total(self):
        return sum(self.penalty_window)

class Solution():
    def __repr__(self):
        return f'(viol_main={self.viol_main},viol_window={self.viol_window}, obj={self.obj},)'

    def __lt__(self, other):
        if self.viol_main != other.viol_main:
            return self.viol_main < other.viol_main
        else:
            if self.viol_window != other.viol_window:
                return self.viol_window < other.viol_window
            else:
                return self.obj < other.obj

    def __len__(self):
        return len(self.visited)

    def __init__(self, node):
        self.visited = []
        self.obj = 0
        self.tonnage_so_far = 0
        self.viol_main = np.inf
        self.viol_window = np.inf
        visited_columns = ['Product_Description', 'Al2O3', 'CaO', 'Fe', 'MgO', 'Mn', 'P', 'S', 'SiO2', 'TiO2']
        self.visited_info = {k:[] for k in visited_columns}
        self.parcel_list = []

        if node is not None:
            self.visited = node.visited.copy()
            self.reclaimed_cuts_keys = node.reclaimed_cuts_keys.copy()


    def clean(self):
        self.visited = []
        visited_columns = ['Product_Description', 'Al2O3', 'CaO', 'Fe', 'MgO', 'Mn', 'P', 'S', 'SiO2', 'TiO2']
        self.visited_info = {k: [] for k in visited_columns}

    def make_parcel(self,node):
        # L=[]
        parcel = Parcel()
        if node is not None:
            L = [node.node_info[x] for x in ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']]
            parcel.penalty_mineral_avg = np.array(L)
        # for k, v in self.visited_info.items():
        #     if k in :
        #         L.append(v[0])
        self.parcel_list.append(parcel)

    def generate_parcel(self, initial_solution, new_parcel):
        for i, p in enumerate(initial_solution.parcel_list):
            parcel = Parcel()
            parcel.penalty_mineral_avg = p.penalty_mineral_avg
            parcel.penalty_main = p.penalty_main
            parcel.penalty_window = p.penalty_window
            parcel.start = p.start
            parcel.end = p.end
            parcel.length = p.length
            parcel.visited = new_parcel[i]
            self.parcel_list.append(parcel)

    def generate_edges(self):
        self.edges = []
        for i in range(len(self.visited)-1):
            self.edges.append((self.visited[i][0].name, self.visited[i+1][0].name))

    def segment(self, start, end):
        return self.visited[start:end]