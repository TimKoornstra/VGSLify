import re
from typing import Dict, List, Optional


class VGSLNode:
    """
    Base class for nodes in the VGSL graph.

    Attributes
    ----------
    name : str
        The identifier or spec of the node.
    next : List[VGSLNode]
        Outgoing edges to other nodes.
    """

    def __init__(self, name: str):
        """
        Parameters
        ----------
        name : str
            The name or spec string of the node.
        """
        self.name = name
        self.next: List[VGSLNode] = []

    def add_edge(self, node: "VGSLNode"):
        """
        Adds a directed edge from this node to another.

        Parameters
        ----------
        node : VGSLNode
            The node to connect as a successor.
        """
        self.next.append(node)


class LayerNode(VGSLNode):
    """
    Represents a regular layer specification in the VGSL graph.

    Attributes
    ----------
    spec : str
        The VGSL layer specification string (e.g., 'C3,3,16').
    """

    def __init__(self, spec: str):
        """
        Parameters
        ----------
        spec : str
            VGSL spec string representing the layer.
        """
        super().__init__(name=spec)
        self.spec = spec


class BranchNode(VGSLNode):
    """
    Represents a branching point in the VGSL graph.

    Attributes
    ----------
    branches : List[VGSLNode]
        List of nodes that start new branches.
    """

    def __init__(self):
        """Initializes a branch node with empty branches."""
        super().__init__(name="branch")
        self.branches: List[VGSLNode] = []

    def add_branch(self, node: VGSLNode):
        """
        Adds a new branch from this node.

        Parameters
        ----------
        node : VGSLNode
            The node where the branch begins.
        """
        self.branches.append(node)
        self.add_edge(node)


class MergeNode(VGSLNode):
    """
    Represents a merge point in the VGSL graph where multiple branches converge.
    """

    def __init__(self):
        """Initializes a merge node."""
        super().__init__(name="merge")


class VGSLGraph:
    """
    Represents the full VGSL graph structure.

    Attributes
    ----------
    entry : Optional[VGSLNode]
        The starting node of the graph.
    exit : Optional[VGSLNode]
        The final node of the graph.
    nodes : List[VGSLNode]
        All nodes in the graph.
    """

    def __init__(self):
        """Initializes an empty VGSLGraph."""
        self.entry: Optional[VGSLNode] = None
        self.exit: Optional[VGSLNode] = None
        self.nodes: List[VGSLNode] = []

    def add_node(self, node: VGSLNode):
        """
        Adds a node to the graph and updates entry/exit points.

        Parameters
        ----------
        node : VGSLNode
            The node to add.
        """
        self.nodes.append(node)
        if self.entry is None:
            self.entry = node
        self.exit = node


def parse_graph_spec(model_spec: str) -> VGSLGraph:
    """
    Parses a VGSL model specification string into a VGSLGraph.

    Handles parentheses for defining branching and merging structures.

    Parameters
    ----------
    model_spec : str
        A space-separated VGSL spec string. Branches are enclosed in parentheses.

    Returns
    -------
    VGSLGraph
        The parsed VGSLGraph.

    Examples
    --------
    >>> g = parse_graph_spec("C3,3,16 (Mp2,2 Fr128) Cr3,3,32")
    >>> [node.name for node in traverse_graph(g.entry)]
    ['C3,3,16', 'branch', 'Mp2,2', 'Fr128', 'merge', 'Cr3,3,32']
    """
    # TODO: Rewrite branching using special token, and each branch within [ ]
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
                if node not in branch_node.branches:
                    branch_node.add_branch(node)
            elif current:
                current.add_edge(node)
            current = node

    return graph


def traverse_graph(entry: VGSLNode) -> List[VGSLNode]:
    """
    Performs a depth-first traversal from the given entry node.

    Ensures each node is visited exactly once.

    Parameters
    ----------
    entry : VGSLNode
        The starting node for traversal.

    Returns
    -------
    List[VGSLNode]
        Nodes in depth-first visit order.
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
