import math

class MonteCarloNode:
    """Class representing a node in the search tree.
    Stores tree search stats for UCB1.
    """
    
    def __init__(self, parent, move, state, unexpanded_moves, is_chance_node=False):
        """Create a new MonteCarloNode in the search tree.
        
        arguments:
        parent -- The parent node.
        move -- Last move played to get to this state.
        state -- The corresponding state.
        unexpanded_moves -- The node's unexpanded child moves.
        is_chance_node -- Indicates whether a random element is to be considered at the node.
        """
        self.move = move
        self.state = state
        
        self.move_num = 0
        self.win_num = 0
        
        self.parent = parent
        self.children = {}
        for move in unexpanded_moves:
            self.children[str(move)] = {"move": move, "node": None}
            
        self.is_chance_node = is_chance_node
    
    def child_node(self, move):
        """Get the MonteCarloNode corresponding to the given play.
        
        arguments:
        move -- The move leading to the child node.
        
        return: The child node corresponding to the move given.
        """
        try:
            child = self.children[str(move)]
        except KeyError:
            raise Exception("No such move!")
        if child["node"] == None:
            raise Exception("Child is not expanded!")
        return child["node"]
    
    def expand(self, move, child_state, unexpanded_moves, is_chance_node=False):
        """Expand the specified child move and return the new child node.
        Add the node to the array of children nodes.
        Remove the move from the array of unexpanded moves.

        arguments:
        move -- The move to expand.
        child_state -- The child state corresponding to the given move.
        unexpanded_plays -- The given child's unexpanded child moves.
        is_chance_node -- Indicates whether a random element is to be considered at the child node.
        
        return: The new child node.
        """
        if str(move) not in self.children:
            raise Exception("No such move!")
        child_node = MonteCarloNode(self, move, child_state, unexpanded_moves, is_chance_node)
        self.children[str(move)] = {"move": move, "node": child_node}
        return child_node
    
    def all_moves(self):
        """Get all legal moves from this node.
        
        return: All moves.
        """
        moves = []
        for child in self.children.values():
            moves.append(child["move"])
        return moves
    
    def unexpanded_moves(self):
        """Get all unexpanded legal moves from this node.
        
        return: All unexpanded moves.
        """
        moves = []
        for child in self.children.values():
            if child["node"] == None:
                moves.append(child["move"])
        return moves
    
    def is_fully_expanded(self):
        """Whether this node is fully expanded.
        
        return: Whether this node is fully expanded.
        """
        for child in self.children.values():
            if child["node"] == None:
                return False
        return True
    
    def is_leaf(self):
        """Whether this node is terminal in the game tree, not inclusive of termination due to winning.
        
        return: Whether this node is a leaf in the tree.
        """
        if len(self.children) == 0:
            return True
        return False
    
    def get_UCB1(self, param):
        """Get the UCB1 value for this node.
        
        arguments:
        param -- The exploration parameter in the UCB1 algorithm.
        
        return: The UCB1 value of this node.
        """
        exploitation_value = self.win_num / self.move_num
        exploration_value = param * (math.log(self.parent.move_num) / self.move_num)**(1/2)
        return exploitation_value + exploration_value