from tkinter import *
from tkinter import simpledialog, messagebox
from game import Game
import players
import numpy as np
import math
from monte_carlo import MonteCarlo
import threading
import copy
from scipy import ndimage
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from game_env import GameEnv

class GUI():
    """Class to visualise the game."""
    RESOLUTION = (1280, 720)
    
    # colors
    PLAYER_TO_MOVE_COLOR = "light green"
    BOARD_COLOR_1 = "green"
    BOARD_COLOR_2 = "orange"
    PLAYER1_COLOR = "white"
    PLAYER2_COLOR = "red"
    POWER_CARD_COLOR = "light blue"
    COMPUTER_THINKING_LABEL_FONT_COLOR = "white"
    BACKGROUND_COLOR = "saddle brown"
    
    # fonts
    SQUARE_FONT = ("Arial", 16)
    PLAYER_LABEL_FONT = ("Arial", 13)
    POWER_CARD_FONT = ("Arial", 16)
    HERO_CARD_USE_CB_FONT = ("Arial", 12)
    COMPUTER_THINKING_LABEL_FONT = ("Arial", 12)
    STACK_FONT = ("Arial", 12)
    
    # symbols
    DIRECTION_SYMBOLS = ["\u2b68","\u2b63","\u2b69","\u2b60","\u2b66","\u2b61","\u2b67","\u2b62"]
    LAST_MOVE_SYMBOL = "O"
    COLOR_CHANGE_SYMBOL = "X"
    
    # sizes
    SQUARE_SIZE = 50
    PLAYER_LABEL_WIDTH = 700
    PLAYER_LABEL_HEIGHT = 20
    POWER_CARD_WIDTH = 50
    POWER_CARD_HEIGHT = 100
    POWER_CARD_INPUT_FIELD_HEIGHT = 30
    HERO_CARD_WIDTH = 100
    HERO_CARD_HEIGHT = 50
    HERO_CARD_CB_WIDTH = 150
    HERO_CARD_CB_HEIGHT = 30
    THINKING_LABEL_WIDTH = 180
    THINKING_LABEL_HEIGHT = 30
    GAP = 10
    
    def __init__(self, with_power_card_input=False):
        """Initialise a new GUI.
        """
        self.window = Tk()
        self.window.title("The Rose King")
        self.window.geometry(f"{self.RESOLUTION[0]}x{self.RESOLUTION[1]}")
        self.window.config(bg=self.BACKGROUND_COLOR)
        
        # Load pictures.
        self.crown_icon = PhotoImage(file="crown_icon.png")
        self.hero_icon = PhotoImage(file="hero_icon.png")
        
        self.window.iconphoto(False, self.crown_icon)
        
        # Create the menu.
        menu = Menu(self.window)
        self.window.config(menu=menu)
        game_menu = Menu(menu)
        menu.add_cascade(label="Game", menu=game_menu)
        new_game_menu = Menu(game_menu)
        game_menu.add_cascade(label="New Game", menu=new_game_menu)
        new_game_menu.add_command(label="Human vs. Human", command=lambda: self.start_new_game(False, False))
        new_game_menu.add_command(label="Human vs. Computer", command=lambda: self.start_new_game(False, True))
        new_game_menu.add_command(label="Computer vs. Human", command=lambda: self.start_new_game(True, False))
        new_game_menu.add_command(label="Computer vs. Computer", command=lambda: self.start_new_game(True, True))
        
        # Create the game board.
        self.squares = []
        for y in range(Game.BOARD_SIZE):
            self.squares.append([])
            for x in range(Game.BOARD_SIZE):
                bg_color = self.BOARD_COLOR_1
                if (x+y) % 2 == 0:
                    bg_color = self.BOARD_COLOR_2
                square = Label(master=self.window,
                               bg=bg_color, borderwidth=1,
                               relief="solid",
                               compound="center",
                               font=self.SQUARE_FONT)
                square.place(x=x*self.SQUARE_SIZE+self.GAP,
                             y=y*self.SQUARE_SIZE+self.GAP,
                             width=self.SQUARE_SIZE,
                             height=self.SQUARE_SIZE)
                self.squares[y].append(square)
        
        self.player_labels = []
        self.power_cards = []
        self.hero_cards = []
        self.hero_card_use_cbs = []
        self.does_use_hero_card = []
        for i in range(Game.PLAYER_NUM):
            # Create labels for the player information.
            player_bg_color = self.PLAYER1_COLOR
            if i == 1:
                player_bg_color = self.PLAYER2_COLOR
            player_label = Label(master=self.window,
                                 text=f"Player {i+1}",
                                 bg=player_bg_color,
                                 font=self.PLAYER_LABEL_FONT,
                                 anchor="w")
            player_label.place(x=self.GAP+Game.BOARD_SIZE*self.SQUARE_SIZE+self.GAP+self.PLAYER_LABEL_HEIGHT+self.GAP,
                               y=self.GAP+i*(self.PLAYER_LABEL_HEIGHT+self.GAP+self.POWER_CARD_HEIGHT+self.GAP),
                               width=self.PLAYER_LABEL_WIDTH,
                               height=self.PLAYER_LABEL_HEIGHT)
            self.player_labels.append(player_label)
            
            # Create buttons for the power cards.
            self.power_cards.append([])
            for j in range(Game.POWER_CARDS_PLACES_NUM):
                power_card = Button(master=self.window,
                                    bg=self.POWER_CARD_COLOR,
                                    font=self.POWER_CARD_FONT,
                                    command=lambda j=j: self.execute_move(j))
                self.power_cards[i].append(power_card)
            
            # Create labels for the hero cards.
            self.hero_cards.append([])
            for j in range(Game.HERO_CARDS_NUM):
                hero_card = Label(master=self.window, bg=player_bg_color, image=self.hero_icon)
                self.hero_cards[i].append(hero_card)
                
            # Create for each player a checkbox so that they can activate a hero card.
            cb_var = BooleanVar()
            hero_card_use_cb = Checkbutton(master=self.window,
                                           text="Use Hero Card",
                                           font=self.HERO_CARD_USE_CB_FONT,
                                           variable=cb_var)
            hero_card_use_cb.place(x=self.GAP+Game.BOARD_SIZE*self.SQUARE_SIZE+self.GAP+Game.POWER_CARDS_PLACES_NUM*(self.POWER_CARD_WIDTH+self.GAP),
                                   y=self.GAP+self.PLAYER_LABEL_HEIGHT+self.GAP+self.HERO_CARD_HEIGHT+self.GAP+i*(self.POWER_CARD_HEIGHT+self.GAP+self.PLAYER_LABEL_HEIGHT+self.GAP),
                                   width=self.HERO_CARD_CB_WIDTH,
                                   height=self.HERO_CARD_CB_HEIGHT)
            self.does_use_hero_card.append(cb_var)
            self.hero_card_use_cbs.append(hero_card_use_cb)
        
        # Create a label to indicate to indicate whose turn it is.
        self.player_to_move_label = Label(master=self.window,
                                          bg=self.PLAYER_TO_MOVE_COLOR)
        
        # Create a label to show that the computer is calculating a move.
        self.computer_thinking_label = Label(master=self.window,
                                             bg=self.window.cget("bg"),
                                             fg=self.COMPUTER_THINKING_LABEL_FONT_COLOR,
                                             text="Computer is thinking...",
                                             font=self.COMPUTER_THINKING_LABEL_FONT)
        
        # Create a button to draw a card from the stack.
        self.stack = Button(master=self.window,
                            bg=self.POWER_CARD_COLOR,
                            text="Draw Card",
                            font=self.STACK_FONT,
                            command=lambda: self.execute_move(None, None))
        self.stack.place(x=self.GAP+Game.BOARD_SIZE*self.SQUARE_SIZE+self.GAP,
                         y=self.GAP+Game.BOARD_SIZE*self.SQUARE_SIZE-self.POWER_CARD_HEIGHT,
                         width=2*self.POWER_CARD_WIDTH,
                         height=self.POWER_CARD_HEIGHT)
        # Create an input field to determine the power card to be drawn.
        self.with_power_card_input = with_power_card_input
        self.power_card_input = StringVar()
        self.power_card_input_field = Entry(master=self.window,
                                      textvariable = self.power_card_input,
                                      bg=self.POWER_CARD_COLOR,
                                      font=self.STACK_FONT)
        if with_power_card_input:
            self.power_card_input_field.place(x=self.GAP+Game.BOARD_SIZE*self.SQUARE_SIZE+self.GAP,
                                    y=self.GAP+Game.BOARD_SIZE*self.SQUARE_SIZE-self.POWER_CARD_HEIGHT-self.GAP-self.POWER_CARD_INPUT_FIELD_HEIGHT,
                                    width=2*self.POWER_CARD_WIDTH,
                                    height=self.POWER_CARD_INPUT_FIELD_HEIGHT)
        
        # Create labels to show the played power cards.
        self.played_power_cards = []
        for i in range(Game.POWER_CARDS_NUM):
            played_power_card = Label(master=self.window,
                                      bg=self.POWER_CARD_COLOR,
                                      font=self.POWER_CARD_FONT)
            self.played_power_cards.append(played_power_card)
        
        self.game = None

        # for rl agent
        #self.game_env = GameEnv(model=2)
        #self.game_env = ActionMasker(self.game_env, self.game_env.valid_action_mask)
        #model_path = "models/trained_models/model2.zip"
        #self.model = MaskablePPO.load(model_path, env=self.game_env)
        
        self.is_player_computer = None
        self.is_game_started = False
        if not with_power_card_input:
            self.start_new_game(False, False)
        self.window.mainloop()
    
    def start_new_game(self, is_player1_computer, is_player2_computer):
        """Start a new game and mark the players as computers if necessary.
        
        arguments:
        is_player1_computer -- Indicates whether player1 is a computer.
        is_player2_computer -- Indicates whether player2 is a computer.
        """
        self.game = Game(with_power_card_input=self.with_power_card_input)
        # for rl agent
        #self.game_env.set_game(self.game)
        
        self.is_game_started = True

        if self.with_power_card_input:
            # Display a dialogue window for each card slot of both players so that the user can define the starting power card.
            position_names = ["first", "second", "third", "forth", "fifth"]
            power_card_indices = [[], []]
            for i in range(Game.PLAYER_NUM):
                for j in range(Game.POWER_CARDS_PLACES_NUM):
                    power_card_to_draw_index = None
                    while(True):
                        power_card_input = simpledialog.askstring("Starting Card Input", f"Enter the {position_names[j]} starting card of player {i+1}.")
                        try:
                            power_card_to_draw_index = self.convert_power_card_input_to_index(power_card_input)
                            # For "King Tactics" comment out the following if-condition and delete the comment marker below it.
                            if power_card_to_draw_index in power_card_indices[0] or power_card_to_draw_index in power_card_indices[1]:
                                raise ValueError()
                            else:
                                break
                            # break
                        except ValueError:
                            messagebox.showerror(title="Card not available",
                                                 message="The entered card does not exist or is no longer in the deck.")
                        except AttributeError:
                            messagebox.showerror(title="Card Input canceled",
                                                 message="You canceled the card input. The game has been cancelled.")
                            self.is_game_started = False
                            return
                    power_card_indices[i].append(power_card_to_draw_index)
            self.game.set_power_cards(power_card_indices)
        
        self.is_player_computer = [is_player1_computer, is_player2_computer]
        self.sync_game()
        
        # If the first player is a computer, a computer move should be made immediately.
        if self.is_player_computer[0]:
            self.execute_move(None, True)

    def sync_game(self):
        """Synchronise the current game data with the graphical display
        so that the user can see the current state
        """
        # Synchronise the game board.
        for y in range(Game.BOARD_SIZE):
            for x in range(Game.BOARD_SIZE):
                self.squares[y][x].config(image="")
                self.squares[y][x].config(text="")
                if self.game.board[y][x] == -1:
                    self.squares[y][x].config(bg=self.PLAYER1_COLOR)
                elif self.game.board[y][x] == 1:
                    self.squares[y][x].config(bg=self.PLAYER2_COLOR)
                else:
                    bg_color = self.BOARD_COLOR_1
                    if (x+y) % 2 == 0:
                        bg_color = self.BOARD_COLOR_2
                    self.squares[y][x].config(bg=bg_color)
        
        # Synchronise the crown position.
        self.squares[self.game.crown_position[0]][self.game.crown_position[1]].config(image=self.crown_icon)
        
        # Show the movement of the crown during the last move.
        if self.game.last_move != None:
            if not self.game.last_move[0]: # not drawing a card
                marking_text = self.LAST_MOVE_SYMBOL
                self.squares[self.game.last_crown_position[0]][self.game.last_crown_position[1]].config(text=marking_text)
                # When a hero card is used, a different symbol should be
                # displayed to indicate that the color of the square has changed.
                if self.game.last_move[1]:
                    marking_text = self.COLOR_CHANGE_SYMBOL
                self.squares[self.game.crown_position[0]][self.game.crown_position[1]].config(text=marking_text)
                
        for i in range(Game.PLAYER_NUM):
            player_to_move_index = self.game.determine_player_index(self.game.player_to_move)
            
            # Synchronise the player information on the player label.
            valuations = self.game.calc_valuations()
            player_text = f"Player {i+1}"
            if self.is_player_computer[i]:
                player_text += " (Com)"
            points_text = f"      Points: {int(valuations[0][i])}"
            field_size_text = f"      Maximum Field Size: {int(valuations[1][i])}"
            pieces_text = f"      Pieces on the Board: {int(valuations[2][i])}"
            self.player_labels[i].config(text=player_text+points_text+field_size_text+pieces_text)
            
            # Synchronise the power cards of each player.
            for j in range(len(self.game.player_power_cards[i])):
                # Cards from a computer player or a player whose turn it is not should not be clickable.
                if player_to_move_index != i or self.is_player_computer[player_to_move_index]:
                    self.power_cards[i][j].config(state=DISABLED)
                else:
                    self.power_cards[i][j].config(state=NORMAL)
                
                # If there is currently no power card in a card slot, no label should appear there.
                if np.all(self.game.player_power_cards[i][j] == 0):
                    self.power_cards[i][j].place_forget()
                else:
                    self.power_cards[i][j].place(x=self.GAP+Game.BOARD_SIZE*self.SQUARE_SIZE+self.GAP+j*(self.POWER_CARD_WIDTH+self.GAP),
                                                 y=self.GAP+self.PLAYER_LABEL_HEIGHT+self.GAP+i*(self.POWER_CARD_HEIGHT+self.GAP+self.PLAYER_LABEL_HEIGHT+self.GAP),
                                                 width=self.POWER_CARD_WIDTH,
                                                 height=self.POWER_CARD_HEIGHT)
                    # On a label for a power card, the move distance should be
                    # displayed as a number and the move direction as a symbol.
                    max_distance = max(abs(self.game.player_power_cards[i][j][0]), abs(self.game.player_power_cards[i][j][1]))
                    direction = self.game.player_power_cards[i][j] / max_distance
                    direction_index = Game.DIRECTIONS.index(tuple(direction))
                    direction_text = str(int(max_distance))+"\n"+self.DIRECTION_SYMBOLS[direction_index]
                    
                    if np.all(self.game.last_drawn_card != None):
                        # If the card was drawn in the last turn, it should be displayed.
                        if np.all(self.game.player_power_cards[i][j] == self.game.last_drawn_card):
                            direction_text += "\nnew"
                    self.power_cards[i][j].config(text=direction_text)
                    
            # Synchronise the number of available hero cards for each player.
            for j in range(Game.HERO_CARDS_NUM):
                if j <= self.game.player_hero_cards_num[i]-1:
                    self.hero_cards[i][j].place(x=self.GAP+Game.BOARD_SIZE*self.SQUARE_SIZE+self.GAP+Game.POWER_CARDS_PLACES_NUM*(self.POWER_CARD_WIDTH+self.GAP)+j*(self.HERO_CARD_WIDTH+self.GAP),
                                                y=self.GAP+self.PLAYER_LABEL_HEIGHT+self.GAP+i*(self.POWER_CARD_HEIGHT+self.GAP+self.PLAYER_LABEL_HEIGHT+self.GAP),
                                                width=self.HERO_CARD_WIDTH,
                                                height=self.HERO_CARD_HEIGHT)
                else:
                    self.hero_cards[i][j].place_forget()
            
            # The checkbox for activating a hero card should always have to be actively selected.
            self.does_use_hero_card[i].set(False)
            # The checkbox of one is deactivated if the player has no more hero cards,
            # the player is a computer or it is not the player's turn.
            if player_to_move_index != i or self.is_player_computer[player_to_move_index] or self.game.player_hero_cards_num[i] == 0:
                self.hero_card_use_cbs[i].config(state=DISABLED)
            else:
                self.hero_card_use_cbs[i].config(state=NORMAL)
            
        # When it is a computer's turn, it should not be possible to click on the stack.
        if self.is_player_computer[player_to_move_index]:
            self.stack.config(state=DISABLED)
        else:
            self.stack.config(state=NORMAL)
        # Display the current number of cards in the stack.
        self.stack.config(text=f"Draw Card\n\n{len(self.game.drawable_power_cards)} cards\navailable")
        self.power_card_input.set("")
        
        # Show all cards in the discard pile.
        for i in range(Game.POWER_CARDS_NUM):
            if i < len(self.game.played_power_cards):
                cards_per_row = Game.POWER_CARDS_NUM//2
                self.played_power_cards[i].place(x=self.GAP+(i%cards_per_row)*(self.POWER_CARD_WIDTH+self.GAP),
                                                 y=self.GAP+Game.BOARD_SIZE*self.SQUARE_SIZE+self.GAP+i//cards_per_row*(self.POWER_CARD_HEIGHT+self.GAP),
                                                 width=self.POWER_CARD_WIDTH,
                                                 height=self.POWER_CARD_HEIGHT)
                max_distance = max(abs(self.game.played_power_cards[i][0]), abs(self.game.played_power_cards[i][1]))
                direction = self.game.played_power_cards[i] / max_distance
                direction_index = Game.DIRECTIONS.index(tuple(direction))
                self.played_power_cards[i].config(text=str(int(max_distance))+"\n"+self.DIRECTION_SYMBOLS[direction_index])
            else:
                self.played_power_cards[i].place_forget()
        
        # If the game is not over yet, mark the player whose turn it is.
        if self.game.is_game_over():
            self.player_to_move_label.place_forget()
        else:
            player_to_move_index = self.game.determine_player_index(self.game.player_to_move)
            self.player_to_move_label.place(x=self.GAP+Game.BOARD_SIZE*self.SQUARE_SIZE+self.GAP,
                                            y=self.GAP+player_to_move_index*(self.PLAYER_LABEL_HEIGHT+self.GAP+self.POWER_CARD_HEIGHT+self.GAP),
                                            width=self.PLAYER_LABEL_HEIGHT,
                                            height=self.PLAYER_LABEL_HEIGHT)
    
    def convert_power_card_input_to_index(self, power_card_input):
        """Convert a string entered by the user into a power card.
        If the string does not have the correct format, throw an exception.

        arguments:
        power_card_input -- A string entered by the user for a power card.
        """
        try:
            values = [int(value) for value in power_card_input.split(',')]
            if len(values) == 2:
                power_card_to_draw = (values[1], values[0]) # Swap here to have x as the first coordinate.
                is_power_card_in_stack = np.all(self.game.drawable_power_cards == power_card_to_draw, axis=1)
                if not np.any(is_power_card_in_stack):
                    raise ValueError()
                else:
                    power_card_to_draw_index = np.where(is_power_card_in_stack)[0][0]
                    return power_card_to_draw_index
            else:
                raise ValueError()
        except ValueError:
            raise ValueError()
        except AttributeError:
            raise AttributeError()
        
    def execute_move(self, power_card_index, is_computermove=False):
        """Executes a move, usually based on the power card that was clicked.
        
        arguments:
        power_card_index -- Specifies the position of the power card to be played.
        is_computermove -- Specifies whether a computer move is to be executed.
        """
        # When the game is not started over, no move can be made.
        if not self.is_game_started:
            messagebox.showerror(title="Game not started",
                                 message="No game has been started yet.\nA new game can be started.")
            return
        if self.game.is_game_over():
            messagebox.showerror(title="Game Over",
                                 message="The game is already over.\nA new game can be started.")
            return
        
        player_to_move_index = self.game.determine_player_index(self.game.player_to_move)
        # The user should be informed via a label when a computer move is calculated.
        # The move is executed in a separate thread so that the gui is not only
        # updated when the computer has moved.
        if is_computermove:
            self.computer_thinking_label.place(x=self.GAP+Game.BOARD_SIZE*self.SQUARE_SIZE+self.GAP+Game.POWER_CARDS_PLACES_NUM*(self.POWER_CARD_WIDTH+self.GAP)+self.HERO_CARD_CB_WIDTH+self.GAP,
                                               y=self.GAP+self.PLAYER_LABEL_HEIGHT+self.GAP+self.HERO_CARD_HEIGHT+self.GAP+player_to_move_index*(self.POWER_CARD_HEIGHT+self.GAP+self.PLAYER_LABEL_HEIGHT+self.GAP),
                                               width=self.THINKING_LABEL_WIDTH,
                                               height=self.THINKING_LABEL_HEIGHT)
            
            computer_thread = threading.Thread(target=self.execute_computer_move)
            computer_thread.start()
            
        else: # A human player makes a move.
            power_card_to_draw_index = None
            if power_card_index == None: # Draw a direction card.
                move = (True, False, None)
                if self.with_power_card_input:
                    # The power card to be drawn is defined here by the string entered in the input field.
                    try:
                        power_card_to_draw_index = self.convert_power_card_input_to_index(self.power_card_input.get())
                    except ValueError:
                        messagebox.showerror(title="Move not possible",
                                         message="The card to be drawn does not exist or is no longer in the deck.")
                        return
                
            else: # Play a power card.
                played_power_card = list(self.game.player_power_cards[player_to_move_index][power_card_index])
                move = (False, self.does_use_hero_card[player_to_move_index].get(), played_power_card)
            
            # Check if the move is possible
            legal_moves = self.game.get_legal_moves(self.game.player_to_move)
            if move in legal_moves:
                self.game.execute_move(move, self.game.player_to_move, power_card_to_draw_index)
                self.sync_game()
            else:
                messagebox.showerror(title="Move not possible",
                                     message="The choosen move is not possible.")
                return
        
            self.handle_consequences()
    
    def execute_computer_move(self):
        """Calculate a move for a computer player.
        """
        move = players.alphabeta(self.game, 7, -math.inf, math.inf, self.game.player_to_move, 30)[1]
        # for rl opponent
        #move = players.rl(self.game, self.game_env, self.model)
        
        self.window.after(0, lambda: self.execute_computer_move_in_main_thread(move))

    def execute_computer_move_in_main_thread(self, move):
        """Auxiliary method for execute_computer_move to display an eventual dialogue window in the main thread.

        arguments:
        move -- The computer's calculated move.
        """
        power_card_to_draw_index = None
        if move != None:
            if move[0]: # The computer wants to draw a card.
                if self.with_power_card_input:
                    # Display a dialogue window so that the user can define the power card to be drawn.
                    while(True):
                        power_card_input = simpledialog.askstring("Card Drawing", "The computer wants to draw a card.\nEnter the card to be drawn.")
                        try:
                            power_card_to_draw_index = self.convert_power_card_input_to_index(power_card_input)
                            break
                        except ValueError:
                            messagebox.showerror(title="Move not possible",
                                                 message="The card to be drawn does not exist or is no longer in the deck.")
                        except AttributeError:
                            messagebox.showerror(title="Card Input canceled",
                                                 message="You canceled the card input. The game has been cancelled.")
                            self.is_game_started = False
                            return
        
        self.game.execute_move(move, self.game.player_to_move, power_card_to_draw_index)
        self.computer_thinking_label.place_forget()
        self.sync_game()
        self.handle_consequences()
    
    def handle_consequences(self):
        """Check whether a special situation has occurred in the game.
        This could either be that only one player can no longer make a move,
        or that the game is over.
        """
        player_to_move_index = self.game.determine_player_index(self.game.player_to_move)
        
        if not self.game.has_legal_moves(self.game.player_to_move):
            winner = self.game.determine_winner()
            if winner == None: # There is no winner yet because the game is not over yet.
                #As the game is not yet over, only one player can no longer make a move,
                # which is why this player must now sit out.
                messagebox.showwarning(title="No move possible",
                                       message=f"Player {player_to_move_index+1} cannot make a move.\nHe must sit out.")
                self.game.execute_move(None, self.game.player_to_move)
                self.sync_game()
            else: # There is one winner (or draw), so the game is over.
                # Determine why the game is over.
                if self.game.playable_pieces_num == 0:
                    end_message = f"The game is over because all {self.game.PIECES_NUM} pieces have been played."
                else:
                    end_message="The game is over because no player can make another move."
                    
                if winner == 0: # Draw
                    winner_message="The game ends in a draw."
                else:
                    differences = self.game.calc_differences(winner)
                    winner_player_index = self.game.determine_player_index(winner)
                    # Determine why the player has won.
                    if differences[0] > 0:
                        winner_message=f"Player {winner_player_index+1} wins because he has more points."
                    elif differences[1] > 0:
                        winner_message=f"Player {winner_player_index+1} wins because he has the largest connected field."
                    else:
                        winner_message=f"Player {winner_player_index+1} wins because he has more pieces on the board."
                
                messagebox.showinfo(title="Game over", message=end_message+"\n"+winner_message)
                return
            
        # Check whether the next player is a computer and if so, make a computer move.
        player_to_move_index = self.game.determine_player_index(self.game.player_to_move)
        if self.is_player_computer[player_to_move_index]:
            self.execute_move(None, True)