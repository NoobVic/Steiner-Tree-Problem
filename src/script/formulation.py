import gurobipy as gp
from gurobipy import GRB

def ILP(graph):
    # set define
    N, E, V = graph
    root = V[0]
    S = [each for each in N if (each not in V)]
    V = V[1:]
    # delete all the arcs that enter the source vertex
    E = [arc for arc in E if not arc[0][1] == root]
    # create the tuple dictionary of arcs
    E_dict = gp.tupledict(E)
    
    # model creation
    m = gp.Model("Steiner")
    
    E_dict_keys = E_dict.keys()
    X_dict = []
    for k in V:
        for arc in E_dict_keys:
            X_dict.append((arc[0], arc[1], k))
    
    # add variables
    x = m.addVars(X_dict,lb=0,vtype=GRB.INTEGER, name='x') # size: |V| * |E|
    y = m.addVars(E_dict_keys,vtype=GRB.INTEGER, name='y') # size: |E|
    
    # set objective value 2.1
    m.setObjective(gp.quicksum(E_dict[i, j] * y[i, j] for (i, j) in E_dict_keys), GRB.MINIMIZE)
    
    # add constraints
    for i in N:
        for k in V:
            # constraint 2.2
            if i == root:
                m.addConstr(x.sum(i,'*',k) - x.sum('*',i,k) == 1)
            elif i == k:
                m.addConstr(x.sum(i,'*',k) - x.sum('*',i,k) == -1)
            else:
                m.addConstr(x.sum(i,'*',k) - x.sum('*',i,k) == 0)
    
    # constraint 2.3
    for i,j,k in X_dict:
        m.addConstr(x[i,j,k] <= y[i,j])
        
    # update the model
    m.update()
    
    # optimize the model
    m.optimize()
    
    # save the optimal solution
    opt_cost = m.objVal
    
    opt_edges = []
    
    for v in m.getVars():
        # save the vertices
        if v.varName.startswith('y') and v.x != 0:
            opt_edges.append((v.varName[2:-1], v.x))
                
    opt_runtime = m.Runtime
    
    return opt_edges, opt_cost, opt_runtime

def LP(graph):
    # set define
    N, E, V = graph
    root = V[0]
    S = [each for each in N if (each not in V)]
    V = V[1:]
    # delete all the arcs that enter the source vertex
    E = [arc for arc in E if not arc[0][1] == root]
    # create the tuple dictionary of arcs
    E_dict = gp.tupledict(E)
    
    # model creation
    m = gp.Model("Steiner")
    
    E_dict_keys = E_dict.keys()
    X_dict = []
    for k in V:
        for arc in E_dict_keys:
            X_dict.append((arc[0], arc[1], k))
    
    # add variables
    x = m.addVars(X_dict,lb=0,vtype=GRB.CONTINUOUS, name='x') # size: |V| * |E|
    y = m.addVars(E_dict_keys,vtype=GRB.CONTINUOUS, name='y') # size: |E|
    
    # set objective value 2.1
    m.setObjective(gp.quicksum(E_dict[i, j] * y[i, j] for (i, j) in E_dict_keys), GRB.MINIMIZE)
    
    # add constraints
    for i in N:
        for k in V:
            # constraint 2.2
            if i == root:
                m.addConstr(x.sum(i,'*',k) - x.sum('*',i,k) == 1)
            elif i == k:
                m.addConstr(x.sum(i,'*',k) - x.sum('*',i,k) == -1)
            else:
                m.addConstr(x.sum(i,'*',k) - x.sum('*',i,k) == 0)
    
    # constraint 2.3
    for i,j,k in X_dict:
        m.addConstr(x[i,j,k] <= y[i,j])
        
    # update the model
    m.update()
    
    # optimize the model
    m.optimize()
    
    # save the optimal solution
    opt_cost = m.objVal


    opt_edges = []
    
    for v in m.getVars():
        # save the vertices
        if v.varName.startswith('y') and v.x != 0:
            opt_edges.append((v.varName[2:-1], v.x))
                
    opt_runtime = m.Runtime
    
    return opt_edges, opt_cost, opt_runtime