import math
import copy
from random import choice
import numpy as np
from compare import compare

def mcts(state, mct, timeout, policy):
    """Search for the best move with the given tree as long as timeout is specified.
    
    arguments:
    state -- The current game state.
    mct -- The tree to perform the Monte Carlo Search.
    
    return: The "best" calculated move.
    """
    mct.run_search(state, timeout)
    move = mct.best_move(state, policy)
    return move

def random(state, player):
    """Choose a random move from the possible moves.
    
    arguments:
    state -- The current game state.
    player -- The player whose turn it is.
    
    return: A random possible move..
    """
    moves = state.get_legal_moves(player)
    if len(moves) != 0:
        return choice(moves)
    return None

def minimax(state, depth, alpha, beta, player, hero_card_discount):
    """Execute the minimax algorithm with alpha-beta pruning for the given depth.
    
    arguments:
    state -- The current game state.
    depth -- Specifies how many moves should be calculated in advance.
    alpha -- minimum possible value.
    beta -- maximum possible value.
    player -- The player whose turn it is.
    hero_card_discount -- The value that is added to the points per hero card.
    
    return: The "best" calculated move.
    """
    if depth == 0 or state.is_game_over():
        return state.calc_heuristic(hero_card_discount, player), None
    
    if player == state.player_to_move:
        value = [-math.inf, -math.inf, -math.inf]
        moves = state.get_legal_moves(player)
        best_move = moves[0]
        
        for move in moves:
            new_state = copy.deepcopy(state)
            new_state.execute_move(move, player)
            new_value = minimax(new_state, depth-1, alpha, beta, player, hero_card_discount)[0]
            
            if compare(new_value, value) == "greater":
                value = new_value
                best_move = move
            if compare(value, beta) == "greater":
                break # beta cutoff
            if compare(value, alpha) == "greater":
                alpha = value
        
        return value, best_move
    
    else:
        value = [math.inf, math.inf, math.inf]
        moves = state.get_legal_moves(-player)
        
        for move in moves:
            new_state = copy.deepcopy(state)
            new_state.execute_move(move, -player)
            new_value = minimax(new_state, depth-1, alpha, beta, player, hero_card_discount)[0]
            
            if compare(new_value, value) == "smaller":
                value = new_value
            if compare(value, alpha) == "smaller":
                break # alpha cutoff
            if compare(value, beta) == "smaller":
                beta = value
        
        return value, None


def expectiminimax(state, depth, alpha, beta, player, hero_card_discount):
    """Execute the expectiminimax algorithm with alpha-beta pruning for the given depth.
    In contrast to the minimax algorithm, consider all possible outcomes for a random event.
    
    arguments:
    state -- The current game state.
    depth -- Specifies how many moves should be calculated in advance.
    alpha -- minimum possible value.
    beta -- maximum possible value.
    player -- The player whose turn it is.
    hero_card_discount -- The value that is added to the points per hero card.
    
    return: The "best" calculated move.
    """
    if depth == 0 or state.is_game_over():
        return state.calc_heuristic(hero_card_discount, player), None
    
    if player == state.player_to_move:
        value = [-math.inf, -math.inf, -math.inf]
        moves = state.get_legal_moves(player)
        best_move = moves[0]
        
        for move in moves:
            new_value = np.array([0, 0, 0], dtype="float64")
            if move == None or not move[0]:
                new_state = copy.deepcopy(state)
                new_state.execute_move(move, player)
                new_value = expectiminimax(new_state, depth-1, alpha, beta, player, hero_card_discount)[0]
            
            else: # Draw a card.
                # Manually draw each card from the pile once and calculate
                # the minimax value for each of the resulting states.
                # Then take the average of all these values.
                drawable_power_cards_num = len(state.drawable_power_cards)
                for i in range(drawable_power_cards_num):
                    new_state = copy.deepcopy(state)
                    new_state.execute_move(move, player, i)
                    new_value += expectiminimax(new_state, depth-1, alpha, beta, player, hero_card_discount)[0]
                new_value /= drawable_power_cards_num
            
            if compare(new_value, value) == "greater":
                value = new_value
                best_move = move
            if compare(value, beta) == "greater":
                break # beta cutoff
            if compare(value, alpha) == "greater":
                alpha = value
        
        return value, best_move
    
    else:
        value = [math.inf, math.inf, math.inf]
        moves = state.get_legal_moves(-player)
        
        for move in moves:
            new_value = np.array([0, 0 ,0], dtype="float64")
            if move == None or not move[0]:
                new_state = copy.deepcopy(state)
                new_state.execute_move(move, -player)
                new_value = expectiminimax(new_state, depth-1, alpha, beta, player, hero_card_discount)[0]
            
            else: # Draw a card.
                # Manually draw each card from the pile once and calculate
                # the minimax value for each of the resulting states.
                # Then take the average of all these values.
                drawable_power_cards_num = len(state.drawable_power_cards)
                for i in range(drawable_power_cards_num):
                    new_state = copy.deepcopy(state)
                    new_state.execute_move(move, -player, i)
                    new_value += expectiminimax(new_state, depth-1, alpha, beta, player, hero_card_discount)[0]
                new_value /= drawable_power_cards_num
            
            if compare(new_value, value) == "smaller":
                value = new_value
            if compare(value, alpha) == "smaller":
                break # alpha cutoff
            if compare(value, beta) == "smaller":
                beta = value
        
        return value, None