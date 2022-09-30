def dfs(node, bfs_dict=None):
    yield node
    if bfs_dict is not None:
        if node.depth in bfs_dict:
            bfs_dict[node.depth].append(node)
        else:
            bfs_dict[node.depth] = [node]

    for child in node.children:
        for c in dfs(child, bfs_dict):
            yield c


def bfs(node, bfs_dict=None):
    if bfs_dict is None:
        bfs_dict = {}
    if len(bfs_dict) == 0:
        list(dfs(node, bfs_dict))
    for nodes in bfs_dict.values():
        for node in nodes:
            yield node


def post_order_traverse(node):
    for child in node.children:
        yield from post_order_traverse(child)
    yield node


# from https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/scatter_gather.py#L12
def _is_namedtuple(obj):
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )


def _is_primitive(obj):
    return not hasattr(obj, "__dict__")


def _is_private(attr):
    return attr.startswith("__")
