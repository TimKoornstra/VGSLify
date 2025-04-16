import re
from typing import Dict, List, Optional


class VGSLNode:
    """
    Base class for nodes in the VGSL graph.
    Each node may have multiple outgoing edges.
    """

    def __init__(self, name: str):
        self.name = name
        self.next: List[VGSLNode] = []

    def add_edge(self, node: "VGSLNode"):
        self.next.append(node)


class LayerNode(VGSLNode):
    """
    Represents a regular layer spec.
    """

    def __init__(self, spec: str):
        super().__init__(name=spec)
        self.spec = spec


class BranchNode(VGSLNode):
    """
    Represents a branching point; splits into multiple paths.
    """

    def __init__(self):
        super().__init__(name="branch")
        self.branches: List[VGSLNode] = []

    def add_branch(self, node: VGSLNode):
        self.branches.append(node)
        self.add_edge(node)


class MergeNode(VGSLNode):
    """
    Represents a merge point where multiple branches converge.
    """

    def __init__(self):
        super().__init__(name="merge")


class VGSLGraph:
    """
    Encapsulates the full graph of the model spec.
    entry: starting node
    exit: final node
    """

    def __init__(self):
        self.entry: Optional[VGSLNode] = None
        self.exit: Optional[VGSLNode] = None
        self.nodes: List[VGSLNode] = []

    def add_node(self, node: VGSLNode):
        self.nodes.append(node)
        if self.entry is None:
            self.entry = node
        self.exit = node


def parse_graph_spec(model_spec: str) -> VGSLGraph:
    """
    Parses a VGSL spec string into a VGSLGraph, handling '(' and ')' for branching.

    Example: "C3,3,16 (Mp2,2 Fr128) Cr3,3,32"
    Yields a graph with a branch: after conv16, two parallel paths (pool->dense), then merge before conv32.
    """
    # Tokenize: split on spaces, but separate parentheses
    tokens = re.findall(r"\(|\)|[^\s()]+", model_spec)

    graph = VGSLGraph()
    stack: List[Dict[str, VGSLNode]] = []
    current: Optional[VGSLNode] = None

    for tok in tokens:
        if tok == "(":
            branch = BranchNode()
            graph.add_node(branch)
            if current:
                current.add_edge(branch)
            stack.append({"branch": branch})
            current = None

        elif tok == ")":
            context = stack.pop()
            merge = MergeNode()
            graph.add_node(merge)
            ends = context["branch"].branches or ([current] if current else [])
            for end in ends:
                if end:
                    end.add_edge(merge)
            current = merge

        else:
            node = LayerNode(spec=tok)
            graph.add_node(node)
            if stack:
                branch_node = stack[-1]["branch"]
                # first or subsequent in this branch
                if node not in branch_node.branches:
                    branch_node.add_branch(node)
            elif current:
                current.add_edge(node)
            current = node

    return graph


def traverse_graph(entry: VGSLNode) -> List[VGSLNode]:
    """
    Depth-first traversal of the VGSLGraph from `entry`, returning nodes in visit order.
    Ensures each node appears once, respecting data-flow dependencies.
    """
    visited = set()
    order: List[VGSLNode] = []

    def dfs(node: VGSLNode) -> None:
        if node in visited:
            return
        visited.add(node)
        order.append(node)
        for child in node.next:
            dfs(child)

    dfs(entry)
    return order
