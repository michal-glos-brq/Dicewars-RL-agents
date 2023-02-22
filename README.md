Implementation of agent playing DiceWars game.

There are two agents implemented.

Ferda is naive with reacitve architecture.

Kokos works as follows:

There is a neural netowrk (Graph convolution neural network) for Q-value estimation from the game states. There is also a BFS kind algorithm with pruning based on the neural-network estimated Q-value. The whole code is focused on speed, so there might be present some "ugly" pieces of code like list comprehension thrown there like so with no assignements (this `[x for x in ...]` turned out to be a lot faster for then `_ = [x for x in ...]` some reason).

The neural network and it's necessary tools are in estimator.py

For working implemantation of this agent and it's environmet, see https://gitlab.com/michal.glos99/sui-projekt
