from core.utils import np
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
            # else:
            #     self.prec_1[direction] = []
    # def find_prec(self):
        # if self.id == '01-01-01-01-01':

class Parcel():
    def __repr__(self):
        return f'(penalty={self.penalty})'
        # return f'(obj={self.obj},viol={self.viol})'

    # def __len__(self):
    #     return len(self.penalty_mineral_avg)

    def __init__(self):
        self.penalty_mineral_avg = np.array([]) # = ['Al2O3', 'CaO', 'Fe', 'MgO', 'Mn', 'P', 'S', 'SiO2', 'TiO2']
        self.penalty_window = np.array([])
        # self.tonnage = 0
        self.penalty = np.inf
        self.length = 0
        self.start = 0
        self.end = 0

class Cut():
    def __repr__(self):
        return f'(cut={self.name})'

    def __init__(self, node):
        self.node = node[0]
        self.name = self.node.name
        self.direction = node[1]
        self.obj = node[2]

class Solution():
    def __repr__(self):
        return f'(obj={self.obj},viol={self.viol})'

    def __lt__(self, other):
        if self.viol != other.viol:
            return self.viol < other.viol
        else:
            return self.obj < other.obj

    def __len__(self):
        return len(self.visited)

    def __init__(self, node):
        self.visited = []
        self.obj = 0
        self.tonnage_so_far = 0
        self.viol = np.inf
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
            parcel.penalty = p.penalty
            parcel.start = p.start
            parcel.end = p.end
            parcel.length = p.length
            parcel.visited = new_parcel[i]
            # parcel.tonnage = 0
            self.parcel_list.append(parcel)

    def generate_edges(self):
        self.edges = []
        for i in range(len(self.visited)-1):
            self.edges.append((self.visited[i][0].name, self.visited[i+1][0].name))

    # def __deepcopy__(self, memo):
    #     copy_object = Solution()
    #     copy_object.visited = self.visited.copy()
    #     copy_object.obj = self.obj
    #     copy_object.tonnage_so_far = self.tonnage_so_far
    #     copy_object.viol = self.viol
    #     copy_object.visited_info = self.visited_info.copy()

        # return copy_object

    def segment(self, start, end):
        return self.visited[start:end]





