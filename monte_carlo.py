from monte_carlo_node import MonteCarloNode
import time
import math
import random
import copy

class MonteCarlo:
    """Class representing the Monte Carlo search tree.
    Handles the four MCTS steps: selection, expansion, simulation, backpropagation.
    Handles best-move selection.
    """
    
    def __init__(self, UCB1_param=2**(1/2)):
        """Create a Monte Carlo search tree.
        
        arguments:
        UCB1_Param -- The exploration parameter in the UCB1 algorithm.
        """
        self.UCB1_param = UCB1_param
        self.nodes = {}
    
    def make_node(self, state):
        """If state does not exist, create dangling node.
        
        arguments:
        state -- The state to make a dangling node for; its parent is set to None.
        """
        if state.hash_value not in self.nodes:
            state = copy.deepcopy(state)
            unexpanded_moves = state.get_legal_moves(state.player_to_move)
            node = MonteCarloNode(None, None, state, unexpanded_moves)
            self.nodes[state.hash_value] = node
    
    def run_search(self, state, timeout=1):
        """From given state, run as many simulations as possible until the time limit, building statistics.
        
        arguments:
        state -- The state to run the search from.
        timeout -- The time to run the simulations for, in seconds.

        return: Search statistics.
        """
        self.make_node(state)
        
        draws = 0
        total_sims = 0
        
        end_time = time.time() + timeout
        while time.time() < end_time:
            node = self.select(state)
            winner = node.state.determine_winner()
            
            if not node.is_leaf() and winner == None:
                node = self.expand(node)
                winner = self.simulate(node)
            self.backpropagate(node, winner)
            
            if winner == 0:
                draws += 1
            total_sims += 1
        
        return {"runtime": timeout, "simulation": total_sims, "draws": draws}
    
    def best_move(self, state, policy="robust child"):
        """From the available statistics, calculate the best move from the given state.
        
        arguments:
        state -- The state to get the best move from.
        policy -- The selection policy for the "best" move.
        
        return: The best move, according to the given policy.
        """
        self.make_node(state)
        
        #If not all children are expanded, not enough information.
        if not self.nodes[state.hash_value].is_fully_expanded():
            print(self.get_stats(state))
            raise Exception("Not enough information!")
        
        node = self.nodes[state.hash_value]
        all_moves = node.all_moves()
        best_move = None
        
        # Most visits (robust child)
        if policy == "robust child":
            max_value = -math.inf
            for move in all_moves:
                child_node = node.child_node(move)
                if child_node.move_num > max_value:
                    best_move = move
                    max_value = child_node.move_num
        
        # Highest winrate (max child)
        elif policy == "max child":
            max_value = -math.inf
            for move in all_moves:
                child_node = node.child_node(move)
                ratio = child_node.win_num / child_node.move_num
                if ratio > max_value:
                    best_move = move
                    max_value = ratio
        
        return best_move
    
    def select(self, state):
        """Phase 1: Selection
        Select until either not fully expanded or leaf node
        
        arguments:
        state -- The root state to start selection from.
        
        return: The selected node.
        """
        node = self.nodes[state.hash_value]
        while node.is_fully_expanded() and not node.is_leaf():
            moves = node.all_moves()
            if node.is_chance_node:
                # No explicit selection, because a random element cannot be influenced.
                random_move = random.choice(moves)
                node = node.child_node(random_move)
            else:
                best_move = None
                best_UCB1 = -math.inf
                for move in moves:
                    child_UCB1 = node.child_node(move).get_UCB1(self.UCB1_param)
                    if child_UCB1 > best_UCB1:
                        best_move = move
                        best_UCB1 = child_UCB1
            
                node = node.child_node(best_move)
            
        return node
    
    def expand(self, node):
        """Phase 2: Expansion
        Of the given node, expand a random unexpanded child node
        
        arguments:
        node - The node to expand from. Assume not leaf.
        
        return: The new expanded child node.
        """
        moves = node.unexpanded_moves()
        move = random.choice(moves)
        
        child_state = copy.deepcopy(node.state)
        if node.is_chance_node:
            child_state.execute_move(node.move, child_state.player_to_move, move)
            child_unexpanded_moves = child_state.get_legal_moves(child_state.player_to_move)
            child_node = node.expand(move, child_state, child_unexpanded_moves)
            self.nodes[child_state.hash_value] = child_node
        
        elif move == None or not move[0]:
            child_state.execute_move(move, child_state.player_to_move)
            child_unexpanded_moves = child_state.get_legal_moves(child_state.player_to_move)
            child_node = node.expand(move, child_state, child_unexpanded_moves)
            self.nodes[child_state.hash_value] = child_node
        
        else: # Draw a direction card and create chance node.
            child_unexpanded_moves = list(range(len(child_state.drawable_power_cards)))
            child_node = node.expand(move, child_state, child_unexpanded_moves, True)
            # Add str(move) because here is no execute_move to update the hash-value.
            self.nodes[child_state.hash_value+str(move)] = child_node
            
        return child_node
    
    def simulate(self, node):
        """Phase 3: Simulation
        From given node, play the game until a terminal state, then return winner
        
        arguments:
        node -- The node to simulate from.
        
        return: The winner of the terminal game state (0 for draw).
        """
        new_state = copy.deepcopy(node.state)
        winner = new_state.determine_winner()
        if node.is_chance_node:
            moves = list(range(len(new_state.drawable_power_cards)))
            move = random.choice(moves)
            new_state.execute_move(node.move, new_state.player_to_move, move)
            winner = new_state.determine_winner()
        while winner == None:
            moves = new_state.get_legal_moves(new_state.player_to_move)
            move = random.choice(moves)
            new_state.execute_move(move, new_state.player_to_move)
            winner = new_state.determine_winner()
        
        return winner
    
    def backpropagate(self, node, winner):
        """Phase 4: Backpropagation
        From given node, propagate plays and winner to ancestors' statistics
        
        arguments:
        node - The node to backpropagate from. Typically leaf.
        winner - The winner to propagate. A draw is ignored.
        """
        while node != None:
            node.move_num += 1
            if node.state.player_to_move == -winner:
                node.win_num += 1
            
            node = node.parent
    
    def get_stats(self, state):
        """Return MCTS statistics for this node and children nodes.
        
        arguments:
        state - The state to get statistics for.
        
        return: The MCTS statistics.
        """
        node = self.nodes[state.hash_value]
        stats = {"move_num": node.move_num, "win_num": node.win_num, "children": []}
        for child in node.children.values():
            if child["node"] == None:
                stats["children"].append({"move": child["move"], "move_num": None, "win_num": None})
            else:
                stats["children"].append({"move": child["move"], "move_num": child["node"].move_num, "win_num": child["node"].win_num})
        
        return stats
