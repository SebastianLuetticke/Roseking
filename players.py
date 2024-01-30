import math
import copy
from random import choice
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker


def rl(state, env, model):
    """Calculate a move for the given agent.

    arguments:
    state -- The current game state.
    env -- The environment in which the agent operates.
    model -- The model for the agent.

    return: The "best" calculated move.
    """
    env.set_game(state)
    obs = env.get_obs()
    mask = env.valid_action_mask()
    action, _ = model.predict(obs, action_masks=mask)
    move = env.get_move_from_action(action)
    return move

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
    
    return: A random possible move.
    """
    moves = state.get_legal_moves(player)
    if len(moves) != 0:
        return choice(moves)
    return None

def minimax(state, depth, player, hero_card_discount):
    """Execute the minimax algorithm with alpha-beta pruning for the given depth.
    
    arguments:
    state -- The current game state.
    depth -- Specifies how many moves should be calculated in advance.
    player -- The player whose turn it is.
    hero_card_discount -- The value that is added to the points per hero card.
    
    return: The "best" calculated move.
    """
    if depth == 0 or state.is_game_over():
        return state.calc_heuristic(hero_card_discount, player, with_inf=True), None
    
    if player == state.player_to_move:
        value = -math.inf
        moves = state.get_legal_moves(player)
        best_move = moves[0]
        
        for move in moves:
            new_state = copy.deepcopy(state)
            new_state.execute_move(move, player)
            new_value = minimax(new_state, depth-1, player, hero_card_discount)[0]
            
            if new_value > value:
                value = new_value
                best_move = move
        
        return value, best_move
    
    else:
        value = math.inf
        moves = state.get_legal_moves(-player)
        
        for move in moves:
            new_state = copy.deepcopy(state)
            new_state.execute_move(move, -player)
            new_value = minimax(new_state, depth-1, player, hero_card_discount)[0]
            
            if new_value < value:
                value = new_value
        
        return value, None

def alphabeta(state, depth, alpha, beta, player, hero_card_discount):
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
        return state.calc_heuristic(hero_card_discount, player, with_inf=True), None
    
    if player == state.player_to_move:
        value = -math.inf
        moves = state.get_legal_moves(player)
        best_move = moves[0]
        
        for move in moves:
            new_state = copy.deepcopy(state)
            new_state.execute_move(move, player)
            new_value = alphabeta(new_state, depth-1, alpha, beta, player, hero_card_discount)[0]
            
            if new_value > value:
                value = new_value
                best_move = move
            if value >= beta:
                break # beta cutoff
            if value > alpha:
                alpha = value
        
        return value, best_move
    
    else:
        value = math.inf
        moves = state.get_legal_moves(-player)
        
        for move in moves:
            new_state = copy.deepcopy(state)
            new_state.execute_move(move, -player)
            new_value = alphabeta(new_state, depth-1, alpha, beta, player, hero_card_discount)[0]
            
            if new_value < value:
                value = new_value
            if value <= alpha:
                break # alpha cutoff
            if value < beta:
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
        return state.calc_heuristic(hero_card_discount, player, with_inf=True), None
    
    if player == state.player_to_move:
        value = -math.inf
        moves = state.get_legal_moves(player)
        best_move = moves[0]
        
        for move in moves:
            new_value = 0
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
            
            if new_value > value:
                value = new_value
                best_move = move
            if value >= beta:
                break # beta cutoff
            if value > alpha:
                alpha = value
        
        return value, best_move
    
    else:
        value = math.inf
        moves = state.get_legal_moves(-player)
        
        for move in moves:
            new_value = 0
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
            
            if new_value < value:
                value = new_value
            if value <= alpha:
                break # alpha cutoff
            if value < beta:
                beta = value
        
        return value, None