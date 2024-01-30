import numpy as np
import gymnasium as gym
from gymnasium import spaces
from game import Game
import random
import players
import math
import copy
from scipy import ndimage


class GameEnv(gym.Env):
    """Class for an environment for a game.
    """
    metadata = {"render_modes": []}

    REWARD_IMPOSSIBLE_MOVE = -1
    REWARD_POSSIBLE_MOVE = 1
    REWARD_WIN = 100
    REWARD_LOST = -100

    def __init__(self, model=2):
        """Initialise a new environment.
        """
        super(GameEnv, self).__init__()
        
        self.action_num = 50
        self.action_space = spaces.Discrete(self.action_num)
        self.model = model
        if self.model in [1,2,3]:
            self.observation_space = spaces.Box(
                low=-5, high=15, shape=(105,), dtype=np.float64
            )
        elif self.model in [4,5,6]:
            self.observation_space = spaces.Box(
            low=-10000, high=10000, shape=(350,), dtype=np.float64
            )
        else:
            raise ValueError("Invalid model!")

        self.game = None
        self.opponent_model = None
        self.opponent_env = None
        self.has_won = None
        self.evaluation = 0

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state.
        
        arguments:
        seed -- The seed that is used to initialize the environment's pseudorandom number generator.
        options -- Additional information to specify how the environment is reset.
        
        return: observations of the initial state, optional information
        """
        super().reset(seed=seed, options=options)
        
        # Initialise a new game.
        self.game = Game()
        self.has_won = None
        # It is randomly selected whether the training model or the opponent starts the game.
        if random.random() < 0.5:
            self.opponent_model_color = -1
            self.execute_move()
        else:
            self.opponent_model_color = 1

        self.evaluation = 0
        
        observation = self.get_obs()
        info = self.get_info()
        
        return observation, info

    def get_obs(self):
        """Calculate the observations for the current state depending on the model used.

        return: observations of the current state
        """
        if self.model in [1,2,3]:
            return self.get_obs_model123()
        elif self.model in [4,5,6]:
            return self.get_obs_model456()
    
    def get_obs_model123(self):
        """Auxiliary method for get_obs()

        return: observations of the current state
        """
        board_obs = list(self.game.board.flatten()*self.game.player_to_move)
        
        crown_obs = list(self.game.crown_position.copy())
        
        own_player_index = self.game.determine_player_index(self.game.player_to_move)
        other_player_index = self.game.determine_player_index(-self.game.player_to_move)
        
        own_new_crown_positions = self.game.crown_position + self.game.player_power_cards[own_player_index]
        # When the empty direction card [0, 0] is considered, the crown position does not change.
        is_new_position = np.any(own_new_crown_positions != self.game.crown_position, axis=1)
        own_new_crown_positions[np.logical_not(is_new_position)] = [crown_obs[0], crown_obs[1]]
        own_new_crown_positions = list(own_new_crown_positions.flatten())
        
        other_new_crown_positions = self.game.crown_position + self.game.player_power_cards[other_player_index]
        # When the empty direction card [0, 0] is considered, the crown position does not change.
        is_new_position = np.any(other_new_crown_positions != self.game.crown_position, axis=1)
        other_new_crown_positions[np.logical_not(is_new_position)] = [crown_obs[0], crown_obs[1]]
        other_new_crown_positions = list(other_new_crown_positions.flatten())
        
        own_hero_cards_obs = self.game.player_hero_cards_num[own_player_index]
        
        other_hero_cards_obs = self.game.player_hero_cards_num[other_player_index]
        
        observations = board_obs + crown_obs + own_new_crown_positions + other_new_crown_positions + [own_hero_cards_obs, other_hero_cards_obs]
        observations = np.array(observations)
        
        return observations

    def get_obs_model456(self):
        """Auxiliary method for get_obs()

        return: observations of the current state
        """
        board = copy.deepcopy(self.game.board)
        evaluated_board, points_obs = self.evaluate_board(board, self.game.player_to_move)
        board_obs = list(evaluated_board.flatten())
        playable_pieces_num_obs = self.game.playable_pieces_num

        crown_pos_obs = list(self.game.crown_position.copy())
        crown_board_obs = np.zeros((Game.BOARD_SIZE, Game.BOARD_SIZE))
        crown_board_obs[self.game.crown_position[0]][self.game.crown_position[1]] = 1
        crown_board_obs = list(crown_board_obs.flatten())

        own_player_index = self.game.determine_player_index(self.game.player_to_move)
        other_player_index = self.game.determine_player_index(-self.game.player_to_move)
        power_card_pos_obs = [[],[]]
        power_card_board_obs = [np.zeros((Game.BOARD_SIZE, Game.BOARD_SIZE)), np.zeros((Game.BOARD_SIZE, Game.BOARD_SIZE))]
        for i in range(2):
            if i == 0:
                player_index = own_player_index
            else:
                player_index = other_player_index
            player = self.game.determine_player(player_index)
            
            new_crown_positions = self.game.crown_position + self.game.player_power_cards[player_index]
            power_card_pos_obs[i] = list(new_crown_positions.flatten())
            is_new_position = np.any(new_crown_positions != self.game.crown_position, axis=1)
            is_new_position_on_board = (new_crown_positions >= 0) & (new_crown_positions < Game.BOARD_SIZE)
            is_new_position_on_board = is_new_position_on_board[:, 0] & is_new_position_on_board[:, 1]

            possible_power_card_indices = np.where(is_new_position_on_board & is_new_position)[0]
            for index in range(Game.POWER_CARDS_PLACES_NUM):
                if index in possible_power_card_indices:
                    tmp_piece = board[new_crown_positions[index][0]][new_crown_positions[index][1]]
                    board[new_crown_positions[index][0]][new_crown_positions[index][1]] = player
                    
                    points_dif = self.evaluate_board(board, self.game.player_to_move)[1] - points_obs
                    if self.game.board[new_crown_positions[index][0]][new_crown_positions[index][1]] == -player:
                        if player == self.game.player_to_move:
                            points_dif -= Game.HERO_CARD_DISCOUNT
                        else:
                            points_dif += Game.HERO_CARD_DISCOUNT
                    
                    board[new_crown_positions[index][0]][new_crown_positions[index][1]] = tmp_piece
                    power_card_board_obs[i][new_crown_positions[index][0]][new_crown_positions[index][1]] = points_dif
            power_card_board_obs[i] = list(power_card_board_obs[i].flatten())
        
        own_hero_cards_obs = self.game.player_hero_cards_num[own_player_index]
        other_hero_cards_obs = self.game.player_hero_cards_num[other_player_index]
        
        observations = board_obs + [points_obs, playable_pieces_num_obs] + crown_pos_obs + crown_board_obs + power_card_pos_obs[0] + power_card_board_obs[0] + power_card_pos_obs[1] + power_card_board_obs[1] + [own_hero_cards_obs, other_hero_cards_obs]
        observations = np.array(observations)
        
        return observations

    def evaluate_board(self, board, player):
        """Calculate how strong a piece placed on the board is.

        arguments:
        board -- The game board on which the pieces are placed.
        player -- The player from whose point of view the strength of the stones is calculated.
        
        return: A game board on which the strengths of the pieces placed are stored.
        """
        new_board = copy.deepcopy(board)

        points = 0
        for current_player in range(-1, 2, 2):
            pieces_of_player = board == current_player
    
            labeled_fields, field_num = ndimage.label(pieces_of_player)
            field_sizes = ndimage.sum(board, labeled_fields, range(1, field_num+1))
            for label, size in zip(range(1, field_num+1), field_sizes):
                points_for_field = size**2
                if current_player == player:
                    new_board[labeled_fields == label] = points_for_field
                    points += points_for_field
                else:
                    new_board[labeled_fields == label] = -points_for_field
                    points -= points_for_field
        
        return new_board, points
    
    def get_info(self):
        """Return additional information about the environment.

        return: additional information
        """
        return {"won": self.has_won}

    def step(self, action):
        """Run one timestep of the environment’s dynamics using the agent action.

        arguments:
        action -- The index of the action that the agent executes.

        return: observations of the next state,
                reward for reaching the next state by executing the action,
                whether the agent reaches a terminal state,
                whether the truncation condition outside the scope of the MDP is satisfied,
                optional information
        """
        move = self.get_move_from_action(action)
        
        possible_moves = self.game.get_legal_moves(self.game.player_to_move)
        reward = 0
        terminated = False
        truncated = False  # no limit for the number of steps here
        if move not in possible_moves:
            reward = self.REWARD_IMPOSSIBLE_MOVE
            terminated = True
        else:
            self.game.execute_move(move, self.game.player_to_move)
            reward, terminated = self.check_game_end()
            if not terminated:
                self.execute_move() # opponent's move
                reward_after_opponent, terminated = self.check_game_end()
                if terminated:
                    reward = reward_after_opponent

        observation = self.get_obs()
        info = self.get_info()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def check_game_end(self):
        """Check whether the current game has ended and calculate the reward for the agent accordingly.

        return: the reward for the agent, whether the game has ended
        """
        reward = 0
        winner = self.game.determine_winner()
        terminated = bool(winner != None)
            
        if winner == -self.opponent_model_color:
            reward = self.REWARD_WIN
            self.has_won = True
        elif winner == self.opponent_model_color:
            reward = self.REWARD_LOST
            self.has_won = False
        else:
            new_evaluation = self.game.calc_heuristic(Game.HERO_CARD_DISCOUNT, -self.opponent_model_color)/10
            if self.model in [3,5,6]:
                reward = new_evaluation - self.evaluation
            self.evaluation = new_evaluation

        if self.model == 1:
            reward = self.REWARD_POSSIBLE_MOVE
        elif self.model == 6:
            # Limit the reward to between -1 and 1.
            reward = reward / 100

        return reward, terminated

    def set_opponent_model(self, opponent_model):
        """Set the model for the opponent.

        arguments:
        opponent_model -- The opponent's model.
        """
        self.opponent_model = opponent_model

    def set_opponent_env(self, opponent_env):
        """Set the environment for the opponent.

        arguments:
        opponent_model -- The environment's model.
        """
        self.opponent_env = opponent_env

    def set_game(self, game):
        """Define the game in which the agent is to participate.

        arguments:
        game -- The game to be played.
        """
        self.game = game

    def execute_move(self):
        """Execute a move for the agent's opponent.
        """
        moves = self.game.get_legal_moves(self.game.player_to_move)
        move_to_play = None
        if moves[0] == None: # Sit out.
            self.game.execute_move(None, self.game.player_to_move)
            return
        if self.opponent_model == None: # random move
            move_to_play = random.choice(moves)
        elif self.opponent_model == "alphabeta":
            move_to_play = players.alphabeta(self.game, 3, -math.inf, math.inf, self.game.player_to_move, 30)[1]
        else:
            self.opponent_env.set_game(self.game)
            obs = self.opponent_env.get_obs()
            mask = self.opponent_env.valid_action_mask()
            action, _ = self.opponent_model.predict(obs, action_masks=mask)
            move_to_play = self.opponent_env.get_move_from_action(action)

        self.game.execute_move(move_to_play, self.game.player_to_move)
    
    def get_move_from_action(self, action):
        """Map the action index to a move.

        return: The move that corresponds to the action index.
        """
        if action == self.action_num - 1:
            return None # Sit out.
        if action == self.action_num - 2:
            return (True, False, None) # Draw a card.
        
        is_using_hero_card = False
        if action > Game.POWER_CARDS_NUM-1:
            is_using_hero_card = True
        
        direction = Game.DIRECTIONS[action%Game.DIRECTIONS_NUM]
        distance = (action%Game.POWER_CARDS_NUM) // Game.DIRECTIONS_NUM + 1
        power_card = [direction[0]*distance, direction[1]*distance]
        
        return (False, is_using_hero_card, power_card)

    def get_action_from_move(self, move):
        """Map the move to an action index.

        return: The action index that corresponds to move.
        """
        if move == None: # Sit out.
            return self.action_num - 1
        if move[0]: # Draw a card.
            return self.action_num - 2

        action = 0
        if move[1]: # Use a hero card.
            action += Game.POWER_CARDS_NUM

        power_card = move[2]
        distance = max(abs(power_card[0]), abs(power_card[1]))
        direction = (power_card[0]/distance, power_card[1]/distance)
        action += (distance-1) * Game.DIRECTIONS_NUM
        action += Game.DIRECTIONS.index(direction)

        return action
    
    def valid_action_mask(self):
        """Create an action mask.

        return: An action mask for the current game state.
        """
        possible_moves = self.game.get_legal_moves(self.game.player_to_move)
        possible_actions = [self.get_action_from_move(move) for move in possible_moves]
        #print(possible_actions)
        action_mask = np.full(self.action_num, False, dtype=bool)
        action_mask[possible_actions] = True
        #print(action_mask)

        return action_mask