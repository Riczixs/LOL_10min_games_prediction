import graphviz as gz
from Tree import Node

dot = gz.Digraph(comment = 'Decision Tree')
def add_nodes(node: Node):
    if node is not None:
        dot.node(str(node.id), node.__str__())
        add_nodes(node.left)
        add_nodes(node.right)
   
def add_edges(node: Node):
    if node.left is not None and node.right is not None:
        dot.edge(str(node.id), str(node.left.id))
        dot.edge(str(node.id), str(node.right.id))
        add_edges(node.right)
        add_edges(node.left)
        
def make_graph(node : Node, size: int, depth: int, index: int):
    global dot
    add_nodes(node)
    add_edges(node)
    dot.render(directory='graphs', filename=f'{size}_{depth}_{index}.gv')
    dot = gz.Digraph(comment = 'Decision Tree')