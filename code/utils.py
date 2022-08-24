import numpy as np
import networkx as nx
import pickle

# class Utility():
#     def __repr__(self):
#         return f'Utility_methods'
#
#     def __init__(self,

class Node():
    def __repr__(self):
        return f'Cut(id={self.index})'

    def __init__(self, row, node_prec_SN, node_prec_NS, node_prec_1_SN, node_prec_1_NS,node_machine , node_row, node_stockpile , node_bench, node_cut):
        self.index = row.Index
        # if row.Index == '01-01-02-04-07':
        #     print('x')# cut_id
        self.product = row.Product_Description # type of product description
        # self.chemical = dict(zip(['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2'],[row.Al2O3, row.Fe, row.Mn, row.P, row.S, row.SiO2])) # Chemical composition of elements in the cut
        # for easier handling we remove the chemical names but through the code we consider their corresponding lower and upper bounds :: ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']
        self.chemical = np.array([row.Al2O3, row.Fe, row.Mn, row.P, row.S, row.SiO2])
        self.cut_tonnage = row.Cut_Tonnage
        self.cost_reclaim = row.Cost
        self.prec = dict(zip(['SN','NS'],[node_prec_SN,node_prec_NS]))
        self.prec_1 = dict(zip(['SN','NS'],[node_prec_1_SN,node_prec_1_NS]))
        self.node_machine, self.node_row, self.node_stockpile, self.node_bench, self.node_cut = node_machine , node_row, node_stockpile , node_bench, node_cut

def parse_cut_id(cut_id: str) -> np.ndarray(shape=(1,5),dtype=int):
    cut_id_separated : list = cut_id.split("-")
    return np.array(cut_id_separated,dtype=np.int8)
# node_machine , node_row, node_stockpile , node_bench, node_cut

def calc_pos_node(node_stockpile : int, node_bench : int, node_cut : int) -> tuple():
    x : int
    y : int
    x = -(node_bench-1)
    y = (10+2)*(node_stockpile-1) + node_cut-1
    return (x,y)

class Solution():
    def __repr__(self):
        return f'Solution(utility={self.total_utility}, penalty_solution={self.total_penalty_solution}, penalty_parcel={self.total_penalty_parcel})'
    def __init__(self):
        self.solution_nodes = [] # all nodes in a solution
        self.solution_cost = [] # cost of reclamation of nodes in the solution
        self.solution_ids = []
        self.solution_penalty = 0 #penalty value of the solution
        self.solution_penalty_parcel = []
        self.solution_tonnage = 0
        self.average_chemical = np.zeros(6) # ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']
        self.solution_path = []

    def __contains__(self, node_id):
        return node_id in self.solution_ids

    def __len__(self):
        return len(self.solution_nodes)

    @property
    def total_utility(self):

        return np.sum(self.solution_cost)

    @property
    def total_penalty_solution(self):

        return self.solution_penalty

    @property
    def total_penalty_parcel(self):

        return np.sum(self.solution_penalty_parcel)

    # def __add__(self, other):
    #     # this function returns a tupe of quality of demands solution
    #     # return (self.total_utility+other.total_utility, self.total_penalty_solution+other.total_penalty_solution, self.total_penalty_parcel+other.total_penalty_parcel)
    #     return (self.total_utility+other[0], self.total_penalty_solution+other[1], self.total_penalty_parcel+other[2])
    def total_fitness(self):
        return (self.total_utility,self.total_penalty_solution, self.total_penalty_parcel)