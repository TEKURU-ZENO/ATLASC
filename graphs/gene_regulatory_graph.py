import networkx as nx

def build_gene_graph(gene_list, interaction_db):
    G = nx.Graph()
    for gene in gene_list:
        G.add_node(gene)
    for edge in interaction_db:
        G.add_edge(edge[0], edge[1])
    return G
