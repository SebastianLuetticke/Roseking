from scipy import ndimage
import numpy as np
from compare import compare

class Game():
    """Class for the game mechanics."""

    BOARD_SIZE = 9 # width and height
    PLAYER_NUM = 2
    PIECES_NUM = 52
    # list of all 8 directions on the board, as (y,x) offsets
    DIRECTIONS = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]
    MAX_DISTANCE = 3
    DIRECTIONS_NUM = 8
    POWER_CARDS_NUM = DIRECTIONS_NUM * MAX_DISTANCE
    POWER_CARDS_PLACES_NUM = 5
    HERO_CARDS_NUM = 4

    def __init__(self):
        """Initialise a new game.
        """
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
        self.playable_pieces_num = self.PIECES_NUM

        # Create all power cards.
        # A power card is a tuple direction*distance.
        self.drawable_power_cards = np.array(
            [(direction[0]*distance, direction[1]*distance)
             for distance in range(1, self.MAX_DISTANCE+1)
             for direction in self.DIRECTIONS])
        np.random.shuffle(self.drawable_power_cards)
        
        # Create the discard pile.
        self.played_power_cards = np.empty((0,2))

        # Give each player five direction cards.
        self.player_power_cards = np.array(
            [self.drawable_power_cards[0:self.POWER_CARDS_PLACES_NUM],
             self.drawable_power_cards[self.POWER_CARDS_PLACES_NUM:self.POWER_CARDS_PLACES_NUM*2]])
        self.drawable_power_cards = self.drawable_power_cards[10:]
        
        # Give each player four hero cards.
        self.player_hero_cards_num = np.array([self.HERO_CARDS_NUM] * self.PLAYER_NUM)

        # Set up the initial crown position.
        self.crown_position = np.array([self.BOARD_SIZE//2] * self.PLAYER_NUM)
        
        self.player_to_move = -1
        
        # Save this information to be able to reconstruct the last move.
        self.last_crown_position = None
        self.last_move = None
        self.last_drawn_card = None
        
        # Create a hash value to uniquely identify game states.
        self.hash_value = str(self.player_power_cards)

    def calc_valuations(self):
        """Calculate for each player the number of points,
        the largest contiguous field and the number of pieces on the board.
        
        return: The number of points, the largest contiguous field and the number of pieces on the board for each player.
        """
        points = [0, 0]
        max_field_sizes = [0, 0]
        pieces_played_num = [0, 0]
        
        for i in range(2):
            player = self.determine_player(i)
            pieces_of_player = self.board == player
            pieces_played_num[i] = np.count_nonzero(pieces_of_player)
            
            # Detect all contiguous fields.
            labeled_fields, field_num = ndimage.label(pieces_of_player)
            for field in range(1, field_num+1):
                field_size = np.count_nonzero(labeled_fields == field)
                points[i] += field_size**2
                if field_size > max_field_sizes[i]:
                    max_field_sizes[i] = field_size
        
        return np.array([points, max_field_sizes, pieces_played_num], dtype="float64")
    
    def calc_differences(self, player):
        """Calculate the differences in the valuations.
        
        arguments:
        player -- The player from whose point of view the point differences are calculated.
        
        return: The differences of the points.
        """
        player_index = self.determine_player_index(player)
        other_player_index = self.determine_player_index(player*(-1))
        
        all_valuations = self.calc_valuations()
        return all_valuations[:, player_index] - all_valuations[:, other_player_index]
    
    def calc_heuristic(self, hero_card_discount, player):
        """Calculate a heuristic function to determine the value
        of a game state for the given player.
        
        arguments:
        hero_card_discount -- The value that is added to the points per hero card.
        player -- The player from whose point of view the heuristic is calculated.
        
        return: The heuristic values for each player.
        """
        player_index = self.determine_player_index(player)
        other_player_index = self.determine_player_index(player*(-1))
        
        differences = self.calc_differences(player)
        # Add a discount value for each hero card still available to increase its value.
        differences[0] += self.player_hero_cards_num[player_index] * hero_card_discount
        differences[0] -= self.player_hero_cards_num[other_player_index] * hero_card_discount
        
        return differences

    def get_legal_moves(self, player):
        """Return all the legal moves for the given player.
        A move is a tupel (is drawing direction card, is using hero card,
        used direction card)
        
        arguments:
        player -- The player whose legal moves are to be calculated.
        
        return: All the legal moves for the given player.
        """
        moves = []
        player_index = self.determine_player_index(player)

        # If all the pieces are on the board, you can no longer make a move.
        if self.playable_pieces_num == 0:
            return [None]
        
        # If you do not have 5 direction cards (at least one place is empty), you can draw a new one.
        if np.any(np.all(self.player_power_cards[player_index] == [0,0], axis=1)):
            moves.append((True, False, None))
        
        # Check whether a direction card can be played.
        new_crown_positions = np.array(self.crown_position + self.player_power_cards[player_index])
        new_crown_positions_num = len(new_crown_positions)
        # When the empty direction card [0, 0] is considered, the crown position does not change.
        is_new_position = np.any(new_crown_positions != self.crown_position, axis=1)
        is_new_position_on_board = (new_crown_positions >= 0) & (new_crown_positions < self.BOARD_SIZE)
        is_new_position_on_board = is_new_position_on_board[:, 0] & is_new_position_on_board[:, 1]
        
        row_indices = new_crown_positions[:, 0].reshape(new_crown_positions_num, 1) % self.BOARD_SIZE
        column_indices = new_crown_positions[:, 1].reshape(new_crown_positions_num, 1) % self.BOARD_SIZE
        # If the array element of the game board is 0, the corresponding square is still unoccupied.
        is_new_position_free = self.board[row_indices, column_indices].T[0] == 0
        
        can_new_position_occupied = is_new_position_on_board & is_new_position_free & is_new_position
        possible_dir_card_indices = np.where(can_new_position_occupied)[0]
        new_moves = [(False, False, self.player_power_cards[player_index][i]) for i in possible_dir_card_indices]
        moves = moves + new_moves
        
        # Check whether a direction card can be played in combination with a hero card.
        if self.player_hero_cards_num[player_index] > 0:
            is_new_position_occupied_by_opponent = self.board[row_indices, column_indices].T[0] == -player
            can_new_position_occupied = is_new_position_on_board & is_new_position_occupied_by_opponent & is_new_position
            possible_dir_card_indices = np.where(can_new_position_occupied)[0]
            new_moves = [(False, True, self.player_power_cards[player_index][i]) for i in possible_dir_card_indices]
            moves = moves + new_moves
                
        if len(moves) == 0:
            return [None]
        return moves

    def has_legal_moves(self, player):
        """Check whether the given player still has valid moves.
        
        arguments:
        player -- Theo player for whom a check for legal moves is to be made.
        
        return: Whether the given player has still valid moves.
        """
        return self.get_legal_moves(player)[0] != None
    
    def is_game_over(self):
        """Check if one of the players still has valid moves.
        If not, the game is over.
        
        return: Whether the game is over.
        """
        return not self.has_legal_moves(1) and not self.has_legal_moves(-1)

    def execute_move(self, move, player, power_card_index=None):
        """Perform the given move for the given color on the board.
        The optional parameter power_card_index can be used to manually draw
        a specific card from the stack.
        
        arguments:
        move -- The move that is to be executed.
        player -- The player who should execute the move.
        power_card_index -- The index of a power card in the stack to be able to draw a specific (and not random) card.
        """
        player_index = self.determine_player_index(player)
        
        self.player_to_move *= -1
        self.hash_value += str(move)
        self.last_crown_position = np.copy(self.crown_position)
        self.last_move = move
        
        if move == None: # The player sits out.
            self.last_drawn_card = None
            return
        
        if move[0]: # Draw a direction card.
            empty_places = np.where(np.all(self.player_power_cards[player_index] == [0,0], axis=1))[0]
            
            # If no specific power card from the stack is provided, a random one is drawn.
            if power_card_index == None:
                power_card_index = np.random.choice(len(self.drawable_power_cards))
            drawn_power_card = self.drawable_power_cards[power_card_index]
            self.last_drawn_card = drawn_power_card
            
            # Add the drawn power card to the player and remove it from the stack.
            self.player_power_cards[player_index][empty_places[0]] = drawn_power_card
            self.drawable_power_cards = np.concatenate((self.drawable_power_cards[0:power_card_index], self.drawable_power_cards[power_card_index+1:]))
            
            # If no more cards can be drawn, then all played cards can be drawn again.
            if len(self.drawable_power_cards) == 0:
                self.drawable_power_cards = self.played_power_cards[:]
                np.random.shuffle(self.drawable_power_cards)
                self.played_power_cards = np.empty((0,2))
            
            self.hash_value += str(drawn_power_card)
            return
        
        # Play a direction card.
        power_card = move[2]
        self.crown_position += power_card
        
        # Add the played power card to the discard pile and remove it from the player.
        self.played_power_cards = np.append(self.played_power_cards, [power_card], axis=0)
        power_card_place = np.where(np.all(self.player_power_cards[player_index] == power_card, axis=1))[0][0] # first index=1 is type of elements
        self.player_power_cards[player_index][power_card_place] = 0
        
        self.board[self.crown_position[0]][self.crown_position[1]] = player
        self.last_drawn_card = None
        
        # A new piece is only placed on the board if no hero card is played,
        # otherwise the existing piece is simply turned over.
        if move[1]:
            self.player_hero_cards_num[player_index] -= 1
        else:
            self.playable_pieces_num -= 1
    
    def determine_winner(self):
        """If the game is over, determine the winner.
        
        return: The player who has won. 0 for a draw, None if the game is not over.
        """
        if not self.is_game_over():
            return None
        
        result = self.calc_differences(-1)
        comparison = compare(result, [0, 0, 0])
        if comparison == "greater": # -1 wins
            return -1
        elif comparison == "smaller": # 1 wins
            return 1
        else: # draw
            return 0
    
    def determine_player_index(self, player):
        """Maps {-1, 1} on {0, 1}.
        
        arguments:
        player -- The player whose array index is to be returned.
        
        return: Th array index for the given player.
        """
        return int(player/2+0.5) 
    
    def determine_player(self, player_index):
        """Maps {0, 1} on {-1, 1}.
        
        arguments:
        player_index -- The array index of the player being searched for.
        
        return: The player for the given array index.
        """
        return int((player_index-0.5)*2)