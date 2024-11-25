# SYSTEM IMPORTS
from collections.abc import Mapping, Sequence
from collections import defaultdict
from typing import Tuple, Type


# PYTHON PROJECT IMPORTS


class LayeredGraph(object):
    NODE_VALUE_TUPLE_INDEX: int = 0
    NODE_PARENT_TUPLE_INDEX: int = 1

    def __init__(self: Type["LayeredGraph"],
                 init_val: float = 0.0
                 ) -> None:
        self.node_layers: Sequence[Mapping[str, Tuple[float, str]]] = list()
        self.init_val: float = init_val

    def add_layer(self: Type["LayeredGraph"]) -> None:
        self.node_layers.append(defaultdict(lambda: tuple([self.init_val, None])))

    def add_node(self: Type["LayeredGraph"],
                 child_node_name: str,
                 child_value: float,
                 parent_node_name: str
                 ) -> None:
        current_layer: Mapping[str, Tuple[float, str]] = self.node_layers[-1]
        current_layer[child_node_name] = tuple([child_value, parent_node_name])

    def get_node_in_layer(self: Type["LayeredGraph"],
                          node_name: str
                          ) -> Tuple[float, str]:
        return self.node_layers[-1][node_name]

