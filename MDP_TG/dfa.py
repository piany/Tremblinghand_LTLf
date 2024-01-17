# -*- coding: utf-8 -*-

from networkx.classes.digraph import DiGraph

from numpy import random

from .mdp import find_MECs, find_SCCs
from .ltl2dra import parse_dra, run_ltl2dra

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class Dfa(DiGraph):
    def __init__(self, statenum, init, edges, aps, acc):
        DiGraph.__init__(self, type='DFA', initial=set(
            [init, ]), accept=acc, symbols=aps)
        print("-------DFA Initialized-------")
        for state in range(1, statenum+1):
            self.add_node(state)
        for (ef, et) in list(edges.keys()):
            guard_string = edges[(ef, et)]
            self.add_edge(ef, et, guard_string=guard_string)
        self.graph['accept'] = acc
        print("-------DFA Constructed-------")
        print(acc)
        print("%s states, %s edges and %s accepting states" %
              (str(len(self.nodes())), str(len(self.edges())), str(len(acc))))

    def check_label_for_dra_edge(self, label, f_dra_node, t_dra_node):
        # ----check if a label satisfies the guards on one dra edge----
        guard_string_list = self[f_dra_node][t_dra_node]['guard_string']
        guard_int_list = []
        for st in guard_string_list:
            int_st = []
            for l in st:
                int_st.append(int(l))
            guard_int_list.append(int_st)
        for guard_list in guard_int_list:
            valid = True
            for k, ap in enumerate(self.graph['symbols']):
                if (guard_list[k] == 1) and (ap not in label):
                    valid = False
                if (guard_list[k] == 0) and (ap in label):
                    valid = False
            if valid:
                return True
        return False

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class Product_Dfa(DiGraph):
    def __init__(self, mdp, dfa):
        DiGraph.__init__(self, mdp=mdp, dfa=dfa, initial=set(),
                         accept=[], name='Product_Dfa')
        self.graph['U'] = mdp.graph['U']
        print("-------Prod DFA Initialized-------")
        self.build_full()

    def build_full(self):
        # ----construct full product----
        for f_mdp_node in self.graph['mdp']:
            for f_dra_node in self.graph['dfa']:
                for f_mdp_label, f_label_prob in self.graph['mdp'].nodes[f_mdp_node]['label'].items():
                    f_prod_node = self.composition(
                        f_mdp_node, f_mdp_label, f_dra_node)
                    for t_mdp_node in self.graph['mdp'].successors(f_mdp_node):
                        mdp_edge = self.graph['mdp'][f_mdp_node][t_mdp_node]
                        for t_mdp_label, t_label_prob in self.graph['mdp'].nodes[t_mdp_node]['label'].items():
                            for t_dra_node in self.graph['dfa'].successors(f_dra_node):
                                t_prod_node = self.composition(
                                    t_mdp_node, t_mdp_label, t_dra_node)
                                truth = self.graph['dfa'].check_label_for_dra_edge(
                                    f_mdp_label, f_dra_node, t_dra_node)
                                if truth:
                                    prob_cost = dict()
                                    for u, attri in mdp_edge['prop'].items():
                                        if t_label_prob*attri[0] != 0:
                                            prob_cost[u] = (
                                                t_label_prob*attri[0], attri[1])
                                    if list(prob_cost.keys()):
                                        self.add_edge(
                                            f_prod_node, t_prod_node, prop=prob_cost)
                                        
        self.build_acc()
        print("-------Prod DFA Constructed-------")
        print("%s states, %s edges and %s accepting states" % (
            str(len(self.nodes())), str(len(self.edges())), str(len(self.graph['accept']))))

    def composition(self, mdp_node, mdp_label, dfa_node):
        prod_node = (mdp_node, mdp_label, dfa_node)
        if not self.has_node(prod_node):
            Us = self.graph['mdp'].nodes[mdp_node]['act'].copy()
            self.add_node(prod_node, mdp=mdp_node,
                          label=mdp_label, dfa=dfa_node, act=Us)
            if ((mdp_node == self.graph['mdp'].graph['init_state']) and
                    (dfa_node in self.graph['dfa'].graph['initial'])):
                self.graph['initial'].add(prod_node)
                print("Initial node added:")
                print(self.graph['initial'])
        return prod_node

    def build_acc(self):
        # ----build accepting pairs----
        accs = []
        for acc_pair in self.graph['dfa'].graph['accept']:
            I = acc_pair[0]  # +set
            Ip = set([prod_n for prod_n in self.nodes() if prod_n[2] in I])
            accs.append([Ip])
        self.graph['accept'] = accs


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class Product_MDPST_dfa:
    def __init__(self, smdp, dfa, PROP): 
        self.name= 'Product_MDPST_dfa'
        self.U = set(smdp.U) 
        self.C = smdp.C
        self.PROP = PROP
        print("-------Prod MDPST Initialized-------")
        self.add_nodes(smdp, dfa)
        self.add_edges(smdp, dfa)
        self.add_state_action_dict(smdp, dfa)
        print("-------Prod MDPST Constructed-------")
        print("%s states, %s edges" % (
            str(len(self.prod_nodes)), str(len(self.prod_edges.keys()))))

    def add_nodes(self, smdp, dfa):
        prod_node = []
        prod_init = []
        for mdp_node in list(smdp.nodes.keys()):
            for mdp_label, label_prob in smdp.nodes[mdp_node].items():
                for dfa_node in dfa.nodes:
                    node = (mdp_node, mdp_label, dfa_node)
                    prod_node.append(tuple(node))
                    if ((mdp_node == smdp.init_state) and
                        (dfa_node in dfa.graph['initial'])):
                        prod_init.append(tuple(node))
        self.prod_nodes = prod_node
        self.prod_init = prod_init
                        
    def add_edges(self, smdp, dfa):
        prod_edges = dict()
        # ----construct full product----
        for (f_mdp_node, t_mdp_node_set) in list(smdp.edges.keys()):
            for f_mdp_label, f_label_prob in smdp.nodes[f_mdp_node].items():
                for f_dfa_node in dfa.nodes:
                    f_prod_node = (f_mdp_node, f_mdp_label, f_dfa_node)
                    for t_mdp_node in t_mdp_node_set:
                        for t_mdp_label, t_label_prob in smdp.nodes[t_mdp_node].items():
                            for t_dfa_node in dfa.successors(f_dfa_node):
                                t_prod_node = (t_mdp_node, t_mdp_label, t_dfa_node)
                                truth = dfa.check_label_for_dra_edge(
                                    f_mdp_label, f_dfa_node, t_dfa_node)
                                if truth:
                                    prob_cost = dict()                          
                                    for u, attri in smdp.edges[(f_mdp_node, t_mdp_node_set)].items():
                                        if attri[0] != 0:
                                            prob_cost[u] = (attri[0], attri[1])
                                    if list(prob_cost.keys()):
                                        prod_edges[(f_prod_node, t_prod_node)]= prob_cost                                 
        self.prod_edges = prod_edges

    def add_state_action_dict(self, smdp, dfa):
        prod_state_action = dict()
        prod_state_action_state = dict()
        # ----construct full product----
        for (f_mdp_node, t_mdp_node_set) in list(smdp.edges.keys()):
            for f_mdp_label, f_label_prob in smdp.nodes[f_mdp_node].items():
                for f_dfa_node in dfa.nodes:
                    f_prod_node = (f_mdp_node, f_mdp_label, f_dfa_node)                    
                    for t_mdp_node in t_mdp_node_set:
                        for t_mdp_label, t_label_prob in smdp.nodes[t_mdp_node].items():
                            for t_dfa_node in dfa.successors(f_dfa_node):
                                truth = dfa.check_label_for_dra_edge(
                                    f_mdp_label, f_dfa_node, t_dfa_node)
                                if truth:
                                    for u, attri in smdp.edges[(f_mdp_node, t_mdp_node_set)].items():
                                        if attri[0] != 0:                      
                                            t_prod_node_set = set()
                                            for t_mdp_node in smdp.state_action[(f_mdp_node, u)]:
                                                t_prod_node = (t_mdp_node, t_mdp_label, t_dfa_node)
                                                prod_state_action_state[(f_prod_node, u, t_prod_node)] = smdp.edges_prob[(f_mdp_node, t_mdp_node)]
                                                t_prod_node_set.add(tuple(t_prod_node))
                                            prod_state_action[(f_prod_node, u)] = t_prod_node_set 
        self.prod_state_action = prod_state_action
        self.prod_state_action_state = prod_state_action_state

#----------------------------------------
#--------------Optimal policy synthesis under LTL--------------------------
def syn_full_plan_dfa(prod_mdpst, mdpst, Acc, SR):
    # ----Optimal plan synthesis, total cost over plan prefix and suffix----
    print("==========[Optimal full plan synthesis start]==========")
    index_prefix, v_new = optimal_plan_prefix(prod_mdpst, mdpst, Acc, SR)
    plan_prefix = index_prefix
    prefix_risk = 1 - v_new[prod_mdpst.prod_init[0]] 
    if plan_prefix:
        print("Best plan prefix obtained, risk %s" %str(prefix_risk))
        return plan_prefix, v_new
    else:
        print("No valid plan found")
        return None, None

def optimal_plan_prefix(prod_mdp, mdp, Acc, SR):
    # ----Synthesize optimal plan prefix to reach accepting MEC or SCC----
    sf = Acc
    Sr = set()
    v_old = dict()
    for node in prod_mdp.prod_nodes:
        if node not in sf:
            Sr.add(node)
            v_old[node] = 0
        else:
            v_old[node] = 1
    print('Number of suffix states: %s' %len(sf))
    print('Number of prefix states: %s' %len(SR))
    for init_node in prod_mdp.prod_init:
        # ---------solve vi------------
        print('-----')
        print('Value iteration for prefix starts now')
        print('-----')
        num_iteration = 0
        num_num = 0
        delta_old = 1
        for num_iteration in range(20):
            print('Number of interation: %s' %num_iteration)
            if delta_old > 0.001:
                v_new, index_prefix, delta_new = optimal_prefix_value_iteration(prod_mdp, mdp, SR, v_old)
                for s in SR:                    
                    v_old[s] = v_new[s]
                for s in sf:
                    v_old[s] = 1
                num_iteration += 1
                num_num += 1
            else:   
                num_iteration += 1 
            
            delta_old = delta_new
            print(delta_old)
                
        print("Prefix Value iteration completed in interations: %s" %num_num)
        print("delta: %s" %delta_new)
    
    # for node in sf:
    #     v_new[node] = 1

    return index_prefix, v_new

def optimal_prefix_value_iteration(prod_mdp, mdp, Sr, v_old):
    num1 = len(Sr)
    U = prod_mdp.U
    PROP = prod_mdp.PROP
    #print(PROP)
    v_new = dict()
    num2 = len(U)
    vlist = [[0] * num2 for _ in range(num1)]
    index = dict()
    delta = 0
    #print(vlist)
    for idx, s in enumerate(Sr):
        for idu, u in enumerate(U):
            if (s, u) in prod_mdp.prod_state_action.keys():
                t_set = prod_mdp.prod_state_action[(s, u)]
                #print(len(t_set))
                for k, t_group in enumerate(t_set):
                    #print(t_group)
                    #print(k)
                    if len(t_set) == 1:
                        pe = 1
                    else:
                        pe = prod_mdp.prod_state_action_state[(s, u, t_group)]
                    #print(pe)
                    fd = t_group[0]
                    #print(fd)
                    dd = t_group[2]
                    #print(dd)
                    if len(fd) > 0:
                        v_group = []
                        for t_node in fd: 
                            #print(t_node)
                            for fl, prob_fl in mdp.nodes[t_node].items(): 
                                #print(fl)
                                node_valid = (t_node, fl, dd)
                                ss = tuple(node_valid) 
                                if ss in list(v_old.keys()): 
                                    if v_old[ss]:                
                                        v_group.append(v_old[ss])
                        #print(v_group)
                        if len(v_group)>0:
                            vlist[idx][idu] += pe*min(v_group)              
    #print(vlist)
    for idx, s in enumerate(Sr):
        v_new[s], index[s] =  max((value, index) for index, value in enumerate(vlist[idx]))
        error = abs(v_new[s] - v_old[s])
        if error > delta:
            delta = error
    # print(v_new)
    # print(index) 
    # print(delta)  

    return v_new, index, delta
