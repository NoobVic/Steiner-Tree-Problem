import pandas as pd
import numpy as np
import re
import os
import networkx as nx
import timeit
from script.formulation import *
from script.pruning import *
from script.features import *

def read_df(file_name):
    df = pd.read_csv("./df/"+file_name+".csv")
    return df

def save_df(df, file_name):
    df.to_csv("./df/"+file_name+".csv", index=False)

def read_graph_directed(filename):
    with open(filename) as f:
        lines = f.readlines()
        arcs = []
        for line in lines:
            if line == '\n': 
                continue
            parts = line.split()
            det = parts[0]
            if det == 'Name':
                name = parts[1]
            elif det == 'Nodes':
                n_vertices = int(parts[1])
            elif det == 'Edges':
                n_edges = int(parts[1])
            elif det == 'E':
                i = int(parts[1])
                j = int(parts[2])
                c = int(parts[3])
                arcij = ((i,j),c)
                arcji = ((j,i),c)
                arcs.append(arcij)
                arcs.append(arcji)
            elif det == 'Terminals':
                n_terminals = int(parts[1])
        vertices = np.arange(1, int(n_vertices)+1)
        vertices = vertices.tolist()
        terminals = np.arange(1, int(n_terminals)+1)
        terminals = terminals.tolist()
        assert(int(n_edges) == len(arcs)/2)
    f.close()
    return [vertices, arcs, terminals]

def read_graph_undirected(filename):
    with open(filename) as f:
        lines = f.readlines()
        edges = []
        for line in lines:
            if line == '\n': 
                continue
            parts = line.split()
            det = parts[0]
            if det == 'Name':
                name = parts[1]
            elif det == 'Nodes':
                n_vertices = int(parts[1])
            elif det == 'Edges':
                n_edges = int(parts[1])
            elif det == 'E':
                i = int(parts[1])
                j = int(parts[2])
                c = int(parts[3])
                edge = ((i,j),c)
                edges.append(edge)
            elif det == 'Terminals':
                n_terminals = int(parts[1])
        vertices = np.arange(1, int(n_vertices)+1)
        vertices = vertices.tolist()
        terminals = np.arange(1, int(n_terminals)+1)
        terminals = terminals.tolist()
        assert(int(n_edges) == len(edges))
    f.close()
    return [vertices, edges, terminals]

def generate_log(filename):
    # v: vertices; c: cost; r: runtime.
    graph = read_graph_directed(filename)
    ilp_v, ilp_c, ilp_r = ILP(graph)
    lp_v, lp_c, lp_r = LP(graph)
    f = open("./log/"+filename+"_log.txt", "wt")
    f.write(str(ilp_r) + "\n")
    f.write(str(ilp_c) + "\n")
    f.write(str(lp_r) + "\n")
    f.write(str(lp_c) + "\n")
    f.write(str(lp_c == ilp_c))
    f.write("\n")   
    for each in ilp_v:
        f.write(str(each)+ "\n")
    f.write("\n")
    for each in lp_v:
        f.write(str(each)+ "\n")

def read_log(filename):
    with open(filename) as f:
        lines = [line.strip() for line in f.readlines()]
        ilp_rt = lines[0]
        ilp_c = lines[1]
        lp_rt = lines[2]
        lp_c = lines[3]
        # lines[4] is whether ilp_c == lp_c
        sols = lines[5:]
        b = sols.index('')
        ilp_sol = [re.sub("[()',]"," ", term).split() for term in sols[:b]]
        lp_sol = [re.sub("[()',]"," ", term).split() for term in sols[b+1:]]
    return {"ilp_rt" : ilp_rt, "ilp_c" : ilp_c, "ilp_sol" : ilp_sol, "lp_rt" : lp_rt, "lp_c" : lp_c, "lp_sol" : lp_sol}

# Check whether the given file has different LP and ILP result
def file_selection(filename):
    with open(filename) as f:
        if f.readlines()[4].startswith("F"): return True
        f.close()
    return False

# Get all selected files 
def get_selected_files():
    filenames = []
    log_filenames = os.listdir("../log/i080")
    filenames = []
    for filename in log_filenames:
        if file_selection("../log/i080/"+filename):
            filenames.append(filename.split('_')[0])
    log_filenames = os.listdir("../log/i160")
    for filename in log_filenames:
        if file_selection("../log/i160/"+filename):
            filenames.append(filename.split('_')[0])
    return filenames

# Get all paths for selected files
def get_paths(filenames, type=None):
    paths = []
    if type == 'df':
        for each in filenames:
            size = each.split('-')[0]
            path = "../df/"+size+"/"+each+'.csv'
            paths.append(path)
    elif type == 'ds':
        for each in filenames:
            size = each.split('-')[0]
            path = "../ds/"+size+"/"+each+'.stp'
            paths.append(path)
    elif type == 'log':
        for each in filenames:
            size = each.split('-')[0]
            path = "../log/"+size+"/"+each+'_log.txt'
            paths.append(path)
    else:
        print('ERROR: Undefined Type: ', type)
    return paths

def split_x_y(df):
    x = df.drop(columns=['Node 1','Node 2','Weight','ILP'])
    y = df['ILP']
    return x, y

def dataframe_generate(ds_filename, log_filename):
    vertices, edges, terminals = read_graph_undirected(ds_filename)
    series = []
    df = pd.DataFrame(columns = ['Node 1', 'Node 2', 'Weight'])
    for edge in edges:
        nodes = edge[0]
        series.append({'Node 1' : nodes[0] , 'Node 2' : nodes[1], 'Weight' : edge[1]})
    df = pd.DataFrame(columns = ['Node 1', 'Node 2', 'Weight'], data=series)
    log = read_log(log_filename)

    # Feature 1: LP Value
    lp = log['lp_sol']
    ilp = log['ilp_sol']
    df['ILP'] = df.apply(
        lambda row : get_LP(row['Node 1'], row['Node 2'], ilp),
        axis=1
    )

    df['LP'] = df.apply(
        lambda row : get_LP(row['Node 1'], row['Node 2'], lp),
        axis=1
    )

    start = timeit.default_timer()
    # Feature 2 & 3: Normailized Weight
    df['Normalized Weight'] = normalize(df['Weight'])
    df['Normalized Weight Std'] = normalize(df['Weight'], True)

    # Feature 4: Variance (TODO)
    df['Variance'] = df['Weight'].var()

    # Feature 5 & 6: Local Rank
    df['Local Rank 1'] = df.apply(
        lambda row : localrank(
            row['Node 1'], row['Node 2'], df[['Node 1', 'Node 2', 'Weight']]),
        axis=1
    )

    df['Local Rank 2'] = df.apply(
        lambda row : localrank(
            row['Node 2'], row['Node 1'], df[['Node 1', 'Node 2', 'Weight']]),
        axis=1
    )


    # Create Graph object
    G = nx.Graph()
    for index, row in df.iterrows():
        G.add_edge(row['Node 1'],row['Node 2'])

    # Feature 7 & 8: Degree Centrality
    d_cen = nx.degree_centrality(G)
    df['Degree Centrality Max'] = df.apply(
        lambda row : max(d_cen[row['Node 1']], d_cen[row['Node 2']]), axis=1
    )

    df['Degree Centrality Min'] = df.apply(
        lambda row : min(d_cen[row['Node 1']], d_cen[row['Node 2']]), axis=1
    )

    # Feature 9 & 10: Betweenness Centrality
    b_cen = nx.betweenness_centrality(G)
    df['Betweenness Centrality Max'] = df.apply(
        lambda row : max(b_cen[row['Node 1']], b_cen[row['Node 2']]), axis=1
    )

    df['Betweenness Centrality Min'] = df.apply(
        lambda row : min(b_cen[row['Node 1']], b_cen[row['Node 2']]), axis=1
    )

    # Feature 11 & 12: Eigenvector Centrality
    e_cen = nx.eigenvector_centrality(G)
    df['Eigenvector Centrality Max'] = df.apply(
        lambda row : max(e_cen[row['Node 1']], e_cen[row['Node 2']]), axis=1
    )

    df['Eigenvector Centrality Min'] = df.apply(
        lambda row : min(e_cen[row['Node 1']], e_cen[row['Node 2']]), axis=1
    )
    stop = timeit.default_timer()

    # Runtime for feature engineering:
    runtime = stop - start + float(log["lp_rt"])

    return df, runtime
    
def get_LP(i,j,lp):
    for each in lp:
        if i == int(each[0]) and j == int(each[1]) : return float(each[2])
        if i == int(each[1]) and j == int(each[0]) : return float(each[2])
    return 0

def get_ILP(i,j,ilp):
    for each in ilp:
        if i == int(each[0]) and j == int(each[1]) : return float(each[2])
        if i == int(each[1]) and j == int(each[0]) : return float(each[2])
    return 0

def normalize(series, isStd=False):
    if isStd:
    # Normalization by Standard Deviation
        miu = series.mean()
        std = series.std()
        result = (series-miu)/std
    else:
    # Default Normailzation
        min = series.min()
        max = series.max()
        delta = max-min
        result = (series-min)/delta
    return result

def localrank(target, node, df):
    # Find all edges connected to the target node
    tmp_df = df.loc[(df['Node 1'] == target) | (df['Node 2'] == target)]
    # Rank them by the weights
    tmp_df = tmp_df.sort_values(by=['Weight'],ignore_index=True)
    # Locate target edge's index
    index = tmp_df.index[
        (tmp_df['Node 1']==node) | (tmp_df['Node 2']==node)].tolist()[0]
    # Return the normalized rank of it
    return index/len(df)

def solve(filename, clf, ds_path, log_path, threshold):
    graph = read_graph_undirected(ds_path)
    mst = get_mst(graph)

    df = dataframe_generate(ds_path, log_path)
    df_pruned_ml = prune_ml(clf, df, threshold)
    df_pruned_lp = prune_lp(df)

    graph_ml = reconstruct(mst, df_pruned_ml, graph[2])
    graph_lp = reconstruct(mst, df_pruned_lp, graph[2])

    print("Problem: ", filename, "Pruning Method: ML")
    sol, obj, rt = ILP(graph_ml)
    print("The OBJ is: ", obj)
    print("The Runtime is: ", rt)
    print("Problem: ", filename, "Pruning Method: LP")   
    sol, obj, rt = ILP(graph_lp)
    print("The OBJ is: ", obj)
    print("The Runtime is: ", rt)

def prune_ml(clf, df, threshold):
    x,y = split_x_y(df)
    y_pred_proba = clf.predict_proba(x)
    y_pred = (y_pred_proba [:,1] >= threshold).astype('int')
    df['Predict'] = y_pred
    df_pruned = df.loc[(df['Predict'] > 0) | (df['LP'] > 0)]
    return df_pruned['Node 1', 'Node 2', 'Weight']

def prune_lp(df):
    df_pruned = df.loc[df['LP'] > 0]
    return df_pruned['Node 1', 'Node 2', 'Weight']

def get_mst(graph):
    return None

def reconstruct(mst, df, terminals):
    vertices, arcs = graph_generate(df)
    graph = (vertices, arcs, terminals)
    return graph

def graph_generate(df):
    vertices = []
    arcs = []
    for index, row in df.iterrows():
        i = row['Node 1']
        j = row['Node 2']
        c = row['Weight']
        if i not in vertices : vertices.append(i) 
        if j not in vertices : vertices.append(j)
        arcs.append(((i,j),c))
        arcs.append(((j,i),c))
    return vertices, arcs

