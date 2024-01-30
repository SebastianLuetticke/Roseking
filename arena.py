from game import Game
import numpy as np
import time
from tqdm import tqdm
import math
import players
from monte_carlo import MonteCarlo
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import gymnasium
from game_env import GameEnv
import os

def mask_fn(env: gymnasium.Env) -> np.ndarray:
    return env.valid_action_mask()

def suggest_move(state, player_to_move, mode, depth=None, hero_card_discount=None, mcts=None, timeout=None, selection_mode=None, env=None, model=None):
    """Suggest a move for the given player type.

    arguments:
    state -- The current game state.
    player_to_move -- The "color" of the player whose turn it is.
    mode -- The type of player.
    further arguments for the player modes

    return: The player's suggested move.
    """
    if mode == "random":
        return players.random(state, player_to_move)
    elif mode == "expectiminimax":
        return players.expectiminimax(state, depth, -math.inf, math.inf, player_to_move, hero_card_discount)[1]
    elif mode == "alphabeta":
        return players.alphabeta(state, depth, -math.inf, math.inf, player_to_move, hero_card_discount)[1]
    elif mode == "minimax":
        return players.minimax(state, depth, player_to_move, hero_card_discount)[1]
    elif mode == "mcts":
        return players.mcts(state, mcts, timeout, selection_mode)
    elif mode == "rl":
        return players.rl(state, env, model)
        
def play_games(num, player1, player2):
    """Play a number of games between the given players.

    arguments:
    num -- The number of games.
    player1 -- The first participating player (dict)
    player2 -- The seconds participating player (dict)
    """
    stats = [0, 0, 0] # [wins player 1, draws, wins player 2]
    time_player1 = 0
    time_player2 = 0
    move_num_player1 = 0
    move_num_player2 = 0
    
    player_to_move = 1
    for i in tqdm(range(num), desc="Play Games"):
        player_to_move *= -1
        game = Game()
        game.player_to_move = player_to_move
        
        while(not game.is_game_over()):
            if game.player_to_move == -1:
                start_time = time.time()
                move = suggest_move(game, -1, player1["mode"], player1["depth"], player1["hero_card_discount"], player1["mcts"], player1["timeout"], player1["selection_mode"], player1["env"], player1["model"])
                time_player1 += time.time() - start_time
                game.execute_move(move, -1)
                move_num_player1 += 1
            else:
                start_time = time.time()
                move = suggest_move(game, 1, player2["mode"], player2["depth"], player2["hero_card_discount"], player2["mcts"], player2["timeout"], player2["selection_mode"], player2["env"], player2["model"])
                time_player2 += time.time() - start_time
                game.execute_move(move, 1)
                move_num_player2 += 1
                
        winner = game.determine_winner()
        if winner == -1:
            stats[0] += 1
        elif winner == 1:
            stats[2] += 1
        else:
            stats[1] += 1
            
    average_time_player1 = time_player1 / move_num_player1
    average_time_player2 = time_player2 / move_num_player2
        
    print(f"Average time player 1: {average_time_player1}")
    print(f"Average time player 2: {average_time_player2}")
    print(f"Wins player1: {stats[0]}")
    print(f"Draws: {stats[1]}")
    print(f"Wins player2: {stats[2]}")

"""
### Create random player ###
player$player_number$ = {
    "mode": "random",
    "depth": None,
    "hero_card_discount": None,
    "mcts": None,
    "timeout": None,
    "selection_mode": None,
    "env": None,
    "model": None
}

### Create minimax player ###
player$player_number$ = {
    "mode": "minimax",
    "depth": 4,
    "hero_card_discount": 30,
    "mcts": None,
    "timeout": None,
    "selection_mode": None,
    "env": None,
    "model": None
}

### Create alphabeta player ###
player$player_number$ = {
    "mode": "alphabeta",
    "depth": 4,
    "hero_card_discount": 30,
    "mcts": None,
    "timeout": None,
    "selection_mode": None,
    "env": None,
    "model": None
}

### Create expectiminimax player ###
player$player_number$ = {
    "mode": "expectiminimax",
    "depth": 4,
    "hero_card_discount": 30,
    "mcts": None,
    "timeout": None,
    "selection_mode": None,
    "env": None,
    "model": None
}

### Create mcts player ###
mcts$player_number$ = MonteCarlo()
player$player_number$ = {
    "mode": "mcts",
    "depth": None,
    "hero_card_discount": None,
    "mcts": mcts$player_number$,
    "timeout": 3,
    "selection_mode": "robust",
    "env": None,
    "model": None
}

### Create reinforcement learning player ###
env$player_number$ = GameEnv(model=2)
env$player_number$ = ActionMasker(env$player_number$, mask_fn)
env$player_number$.reset()
model_path$player_number$ = "models/trained_models/model2.zip"
model$player_number$ = MaskablePPO.load(model_path$player_number$, env=env$player_number$)
player$player_number$ = {
    "mode": "rl",
    "depth": None,
    "hero_card_discount": None,
    "mcts": None,
    "timeout": None,
    "selection_mode": None,
    "env": env$player_number$,
    "model": model$player_number$
}
"""