from parse import read_input_file, write_output_file
import os
from leaderboard import get_best_outputs
import pandas as pd
def solve(tasks):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing  
    """
    get_best_outputs()


# Here's an example of how to run your solver.
if __name__ == '__main__':
    # for input_path in os.listdir('inputs/'):
    #     output_path = 'outputs/' + input_path[:-3] + '.out'
    #     tasks = read_input_file(input_path)
    #     output = solve(tasks)
    #     write_output_file(output_path, output)
    solve(tasks=None)