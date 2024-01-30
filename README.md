# Roseking

This is the repository for the bachelor thesis "Implementierung und Vergleich verschiedener KI-Ansätze für das Spiel 'Rosenkönig'" ("Implementation and comparison of different AI approaches for the game 'Rose King'") by Sebastian Lütticke. It contains all the files required to produce the results presented in the thesis.  
The objective of this work was initially to implement a "Rose King" version with which a user can easily interact. AI agents were then developed using either Minimax-based algorithms, Monte Carlo Tree Search (MCTS) or a reinforcement learning (RL) approach.  
All AI agents could be tested and further developed through this repository.  
To train and test the developed reinforcement learning agents, make sure that the packages [Stable Baselines3](https://pypi.org/project/stable-baselines3/), [Stable Baselines3 contrib](https://sb3-contrib.readthedocs.io/en/master/guide/install.html) and [Gymnasium](https://pypi.org/project/gymnasium/) are installed.

## Python-Files
- [game.py](game.py): All rules of the game "Rose King" are implemented here.
- [gui.py](gui.py): The graphical user interface is implemented here. The default AI opponent is an alpha-beta agent with a search depth of seven through the code.
- [start_gui.py](start_gui.py): Execute this file to start a new game against another person or an AI via the GUI.
- [start_gui_with_power_card_input.py](start_gui_with_power_card_input.py): Execute this file to test the preset AI agent via the GUI against an AI from another project. Important: If the AI from the "King Tactics" application is to be tested, the lines specified in the gui.py file must be commented out or uncommented.
- [compare.py](compare.py): Contains a method to compare all values of both players, which are necessary to determine the winner of "Rose King".
- [monte_carlo.py](monte_carlo.py): The functions of the MCTS are implemented here.
- [monte_carlo_node,py](monte_carlo_node,py): Here the class for a node in a Monte Carlo tree is implemented.
- [players.py](players.py): The Minimax-based algorithms and methods for accessing the MCTS and an RL agent are implemented here.
- [game_env.py](game_env.py): Here is a Gymnasium-Environment for the "Rose King"-version from game.py implemented.
- [train_model.py](train_model.py): Execute this file to train one of the developed RL models for the agent.
- [arena.py](arena.py): Execute this file to let different AI agents from this project compete against each other without a GUI. The file contains ready-made code sections to create the desired players. Just replace `$player_number$` with a player number that only one player can have. For example, you can simply use 1 and 2 for two players.

## RL Models
The RL models for the best trained RL agents from this work are stored in the [models/trained_models](models/trained_models) folder. When training and testing an RL agent, it is important to always specify the appropriate model.  
The following models were developed in this work:

| Model | Uses Action masking |      Input coding scheme     |                                     Reward mechanism                                     |  
| :---: | :-----------------: | :--------------------------: | ---------------------------------------------------------------------------------------- |  
|   1   |         no          | coding scheme 1 (150 inputs) | +1 for valid move, -1 for invalid move                                                   |  
|   2   |         yes         | coding scheme 1 (150 inputs) | reward at the end of the game                                                            |  
|   3   |         yes         | coding scheme 1 (150 inputs) | reward at the end of the game + reward for each move                                     |  
|   4   |         yes         | coding scheme 2 (350 inputs) | reward at the end of the game                                                            |  
|   5   |         yes         | coding scheme 2 (350 inputs) | reward at the end of the game + reward for each move                                     |  
|   6   |         yes         | coding scheme 2 (350 inputs) | reward at the end of the game + reward for each move, limited to the interval \[-1, +1\] |

As model 1 is not capable of executing only valid moves, it cannot be used to play a real game. 
