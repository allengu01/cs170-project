# CS 170 Project Fall 2021

## Solvers
- knapsack.py: Uses a heuristic to generate the initial task ordering, then uses bottom up DP to generate the best score using that ordering. The output is created through a knapsack-like algorithm.
- altered_knapsack.py: Tries the knapsack solution using randomly generated heuristics and repeats for many trials, taking the best one.
- better_greedy.py: Greedy approach that takes the locally optimal task each iteration.
- basic_genetic.py: Uses a genetic algorithm that starts from the best output found so far and attempts to improve it.
- random_swapping.py: Tries two different methods of swapping (1 - swaps two indices, 2 - swaps a task to a given index, pushing the others backwards). This was used to optimize a solution once it was found with the other algorithms.
- score.py: Computes scores for outputs of a given algorithm.
- leaderboard.py: Set of functions to maintain a leaderboard of all outputs given by the algorithms tried thus far. Generates the set of final inputs by looking at the best algorithm for each input and combining them.
- parse.py: Used to parse input and output files (added a couple more helper functions compared to the skeleton code).

## Generating Outputs
1. Run any of the algorithms to generate a set of outputs. This will be stored in a folder called all_outputs/{algorithm name}
2. Score the outputs by running the file score.py. Within the file, make sure to change the solver_name variable to the name of the algorithm
3. Repeat steps 1 and 2 for all of the algorithms you want to run.
4. Run the leaderboard.py file. This script generates a leaderboard and set of rankings for each input. Then, it takes the best ranked algorithm and combines its inputs into the outputs folder.