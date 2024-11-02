# SYSTEM IMPORTS
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Tuple, Type
import itertools
import math
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from grammar import PCFG
from trees import Tree, Node


UNKNOWN_TERMINAL: str = "<unk>"


class Parser(object):
    def __init__(self: Type["Parser"]) -> None:
        self.grammar: PCFG = PCFG()

    def __str__(self: Type["Parser"]) -> str:
        return "%s" % self.grammar

    def __repr__(self: Type["Parser"]) -> str:
        return "%s" % self

    def _update_grammar_dfs(self: Type["Parser"],
                            n: Node                 # the root of the subtree we are updating the grammar with
                            ) -> None:
        # TODO: Complete me!
        # This method should, given a node, look at its label (either a nonterminal or a terminal symbol)
        # as well as the its children.
        # If the children are empty, then you don't have to do anything
        # If the children are nonempty, you should add the rule to the grammar (label -> children)
        # There is a method in the PCFG class called "add_rule" that you will find useful for doing so.
        # You will also have to iterate/recurse on the children of the node to keep adding rules to the grammar.

        # self.grammar.add_rule(n.label, tuple(child.label for child in n.children))
        # for child in n.children:
        #     self._update_grammar_dfs(child)

        if len(n.children) == 0:
            return
        else:
            self.grammar.add_rule(n.label, tuple(child.label for child in n.children))
            for child in n.children:
                self._update_grammar_dfs(child)




    def _update_grammar_with_tree(self: Type["Parser"],
                                  t: Tree
                                  ) -> None:
        if t is not None:
            self._update_grammar_dfs(t.root)

    def get_start(self: Type["Parser"]) -> str:
        return self.grammar.start

    def _train(self: Type["Parser"],
               tree_strings: Iterable[str],
               already_cnf: bool = False
               ) -> None:
        for tree_string in tree_strings:

            t: Tree = Tree.from_str(tree_string)
            if not already_cnf:
                t.binarize()
                t.remove_unit()

            self._update_grammar_with_tree(t)

        self.grammar._validate_self()

    def train_from_file(self: Type["Parser"],
                        fpath: str,
                        already_cnf: bool = False
                        ) -> None:
        with open(fpath, "r") as f:
            self._train(f, already_cnf=already_cnf)

    def train_from_raw(self: Type["Parser"],
                       tree_strings: Iterable[str],
                       already_cnf: bool = False
                       ) -> None:
        self._train(tree_strings, already_cnf=already_cnf)

    def finalize(self: Type["Parser"],
                 start_nonterm: str
                 ) -> None:
        self.grammar.set_start(start_nonterm)
        self.grammar.normalize_joint()

    def _check_grammar(self: Type["Parser"]) -> None:
        if self.grammar.start is None:
            raise Exception("Grammar not finalized. Call self.finalize(<start_nonterminal>)")

    def _preprocess_sentence(self: Type["Parser"],
                             w: str
                             ) -> Sequence[str]:
        # parse the sentence to remove entries...all of the words are terminals
        list_of_words = w.split()  # this takes care of all extra whitespaces
        for i, word in enumerate(list_of_words):
            if word not in self.grammar.terminals:
                list_of_words[i] = UNKNOWN_TERMINAL
        return list_of_words

    def _traverse_backptrs_dfs(self: Type["Parser"],
                            backptrs: object,
                            cur_row: int,
                            cur_col: int,
                            cur_symb: str
                            ) -> Node:
        node = Node(cur_symb, [])

        if cur_row == 0:
            last_node = Node(backptrs[cur_row][cur_col][cur_symb], [])
            node.append_child(last_node)
            return node
        else:
            l_child_symb, l_row, l_col, r_child_symb, r_row, r_col = backptrs[cur_row][cur_col][cur_symb]
            left_child = self._traverse_backptrs_dfs(backptrs, l_row, l_col, l_child_symb)
            if r_child_symb is not None:
                right_child = self._traverse_backptrs_dfs(backptrs, r_row, r_col, r_child_symb)
            node.append_child(left_child)
            if r_child_symb is not None:
                node.append_child(right_child)
            return node


    def generate_best_tree(self: Type["Parser"],
                           backptrs: object         # your own structure datatype goes here
                           ) -> Tree:
        # TODO: complete me!
        # Here is the main strategy for this method:
        #    lookup the position of the start nonterminal from the backptrs and walk the backptrs down to the leaves
        #    you can do the walking part with self._traverse_backptrs_dfs
        #    but you are free to use whatever indexing you want.

        # root: Node = None
        # return Tree(root)
        root = self.best_nonterm
        n = self.n
        best_tree = self._traverse_backptrs_dfs(backptrs, n-1, 0, root)
        return Tree(best_tree)





    def _cky_traverse(self: Type["Parser"],
                      list_of_words: Sequence[str],
                      update_func_ptr: Callable[[Tuple[int, int],       # coordinate of the target cell (in chart(s))
                                                 Tuple[int, int],       # coordinate of one cell c_ik (in chart(s))
                                                 Tuple[int, int]],      # coordinate of one cell c_jk (in chart(s))
                                                 None]
                      ) -> None:
        # TODO: complete me!
        # This method should only worry about traversing the cells in the cky-algorithm. The structure of the cky
        # algorithm is that when we are at a "target cell" (i.e. a cell to update) in our chart, we will search
        # through a combination of cell pairs. Given a cell pair c_ik, and c_kj, both of which contain a collection
        # of nonterminals, we will search for any rule in our grammar X -> A B where A \in c_ik, and B \in c_kj. If
        # such a rule exists, we will add nonterminal X to the "target cell".

        # Whether we're doing the vanilla cky algorithm or cky_viterbi, the cell traversal structure is the same.
        # So this method will solely be responsible for doing the traversal. Once your code has determined
        # the coordinates of the "target cell" and a cell pair, you should call the update_func_ptr with these
        # cell coordinates.

        # The purpose of the update_func_ptr is to abstract the traversal away from actually updating your
        # dynamic programming structures. In vanilla cky, this is updating a chart. In cky_viterbi, this is
        # updating multiple charts (one for the backpointers and another for the logprobs...you can combine these
        # into a single chart if you want). Either way, the updating functionality is only possible once you know
        # the coordinates. So we can abstract away vanilla cky and cky_viterbi into a cky_traversal on a specific
        # chart(s) updating scheme. The function pointer here is where you will update your chart(s) and only needs
        # to be called from here.

        n = len(list_of_words)
        chart_size = n
        # chart: Sequence[Sequence[Set[str]]] = [[set() for _ in range(chart_size)]
        #                                         for _ in range(chart_size)]
        for row in range(1, chart_size + 1):
            for col in range(n - row + 1):
                for leftRow in range(0, row):
                    rightRow = row - leftRow - 1
                    leftCol = col
                    rightCol = col + leftRow + 1
                    update_func_ptr((row, col), (leftRow, leftCol), (rightRow, rightCol))
        



    def cky(self: Type["Parser"],
            w: str                  # a sentence (not split)
            ) -> Tuple[bool, float]:

        # make sure the grammar is valid
        self._check_grammar()

        # split the sentence into tokens
        list_of_words: Sequence[str] = self._preprocess_sentence(w)

        

        # Since we have a cky_traversal algorithm, the vanilla cky algorithm can be implemented as calling
        # the traversal with a specific function pointer that updates a chart. You will of course have to initialize
        # this chart in this method. I would recommend creating a nested function inside of this method
        # that performs the vanilla cky chart update, and then call cky_traverse with that function pointer.


        # I am choosing to make only the cells that we need. I know the description of the algorithm says to create
        # a nxn chart, but we will only use half of it, so I am only going to create the cells needed.
        # I am defining my chart to be indexed using [row][col] which returns a set of nonterminals
        # I am also assigning row=0 to be the "bottom" of the chart (i.e. the layer that produces terminals)
        # and row=-1 to be the "top" of the chart (where the start nonterminal should be).
        # You are welcome to change this indexing if you want, and you are also welcome to change this paradigm
        # and allocate a full nxn chart if you wish. If you do so this code will need to change.
        chart: Sequence[Sequence[Set[str]]] = [[set() for _ in range(len(list_of_words) + 1)]
                                                for i in range(len(list_of_words) + 1)]

        # initialize chart
        for i, word in enumerate(list_of_words):
            for nonterm, _ in self.grammar.get_rules_to(word):
                chart[0][i].add(nonterm)        # if you change the indexing scheme you need to change this row=0

        def update_cky_chart(target_coords: Tuple[int, int],
                             left_prod_coords: Tuple[int, int],
                             right_prod_coords: Tuple[int, int]
                             ) -> None:
            tr, tc = target_coords
            lr, lc = left_prod_coords
            rr, rc = right_prod_coords
            for lprod, rprod in itertools.product(chart[lr][lc], chart[rr][rc]):
                for nonterm, _ in self.grammar.get_rules_to(lprod, rprod):
                    chart[tr][tc].add(nonterm)

        self._cky_traverse(list_of_words, update_cky_chart)

        # return whether the parse is possible. You don't need to change the second argument from -math.inf.
        # if you change the indexing you will need to change the [row=-1][col=-1]
        return self.grammar.start in chart[len(list_of_words)-1][0], -math.inf


    def cky_viterbi(self: Type["Parser"],
                    w: str,                     # a sentence (not split)
                    log_base: float = 10        # the base of the logprob to use
                    ) -> Tuple[str, float]:
        # make sure the grammar is valid
        self._check_grammar()

        # split the sentence into tokens
        list_of_words: Sequence[str] = self._preprocess_sentence(w)

        # TODO: complete me!
        # Since we have a cky_traversal algorithm, just like the vanilla cky algorithm we can implement this
        # with a nested function that performs the cky_viterbi chart(s) update(s). You will need to initialize
        # your chart(s) in this method and then call the cky_traversal method with the nested function pointer.

        # Once cky_traversal has completed, you will then need to call generate_best_tree to, using your backpointer
        # structure, extract the Tree with the highest logprob

        # return the Tree and the logprob
        # return None, -math.inf

        n = len(list_of_words)
        
        self.n = n

        chart: Sequence[Sequence[Mapping[str, float]]] = [[{} for _ in range(len(list_of_words) + 1)]
                                                              for i in range(len(list_of_words) + 1)]
        backptrs: Sequence[Sequence[Mapping[str, Tuple[int, int, int, int]]]] = [[{} for _ in range(len(list_of_words) + 1)]
                                                                for i in range(len(list_of_words) + 1)]
        

        # initialize chart
        for i, word in enumerate(list_of_words):
            # if word == "'s":
            #     for nonterm, i in [('VBZ', 0.5), ('POS', 0.5)]:
            #         prob = math.log(0.5, i)
            #         if nonterm not in chart[0][i] or prob > chart[0][i][nonterm]:
            #             chart[0][i][nonterm] = prob
            #             backptrs[0][i][nonterm] = word

            for nonterm, prob in self.grammar.get_rules_to(word):
                prob = math.log(prob, log_base)
                if nonterm not in chart[0][i] or prob > chart[0][i][nonterm]:
                    chart[0][i][nonterm] = prob
                    backptrs[0][i][nonterm] = word


        def update_cky_viterbi_chart(target_coords: Tuple[int, int],
                                    left_prod_coords: Tuple[int, int],
                                    right_prod_coords: Tuple[int, int]
                                    ) -> None:
            tr, tc = target_coords
            lr, lc = left_prod_coords
            rr, rc = right_prod_coords

            # only left_prob_coords
            for lprod, lprod_prob in chart[lr][lc].items():
                for nonterm, rule_prob in self.grammar.get_rules_to(lprod):
                    print("lprod", lprod, "nonterm", nonterm, "rule_prob", rule_prob)
                    if rule_prob == 0:
                        rule_prob = -math.inf
                    else:
                        rule_prob = math.log(rule_prob, log_base)
                    prob = lprod_prob + rule_prob
                    if nonterm not in chart[tr][tc] or prob > chart[tr][tc][nonterm]:
                        chart[tr][tc][nonterm] = prob
                        backptrs[tr][tc][nonterm] = (lprod, lr, lc, None, -1, -1)


            for lprod, rprod in itertools.product(chart[lr][lc].items(), chart[rr][rc].items()):
                lprod_nonterm, lprod_prob = lprod
                rprod_nonterm, rprod_prob = rprod

                for nonterm, rule_prob in self.grammar.get_rules_to(lprod_nonterm, rprod_nonterm):
                    if rule_prob == 0:
                        rule_prob = -math.inf
                    else:
                        rule_prob = math.log(rule_prob, log_base)
                    prob = lprod_prob + rprod_prob + rule_prob
                    if nonterm not in chart[tr][tc] or prob > chart[tr][tc][nonterm]:
                        chart[tr][tc][nonterm] = prob
                        backptrs[tr][tc][nonterm] = (lprod_nonterm, lr, lc, rprod_nonterm, rr, rc)


        self._cky_traverse(list_of_words, update_cky_viterbi_chart)

        # return the best parse and its log probability
        if chart[n-1][0] == {}:
            print("No parse found")
            for i in range(n):
                print(chart[i])
            return None, -math.inf
        best_items = chart[n-1][0].items()
        self.best_nonterm, self.best_logprob = max(best_items, key=lambda item: item[1])

        best_tree = self.generate_best_tree(backptrs)
        # best_tree.restore_unit()
        # best_tree.unbinarize()

        # for str, float in self.grammar.get_rules_to("NP"):
        #     print(str, float)

        # for nonterm, prob in self.grammar.get_rules_to("NP_NNS", "PP"):
        #     print(nonterm, prob)

        return best_tree, self.best_logprob
