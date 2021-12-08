from parse import read_input_file, read_output_file, read_best_output_file, write_output_file
from leaderboard import save_scores
import os

def fix(tasks, order):
    t = 0
    output = []
    for i in order:
        task = tasks[i - 1]
        if t + task.get_duration() > 1440:
            continue
        output.append(task.get_task_id())
        t += task.get_duration()
    return output

def solve(tasks, order=[]):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing  
    """
    return fix(tasks, order)


for size in os.listdir('inputs/'):
    if size not in ['small', 'medium', 'large']:
        continue
    for input_file in os.listdir('inputs/{}/'.format(size)):
        if size not in input_file:
            continue
        input_path = 'inputs/{}/{}'.format(size, input_file)
        output_path = 'outputs/{}/{}.out'.format(size, input_file[:-3])
        print(input_path, output_path)
        tasks = read_input_file(input_path)
        best_output = read_best_output_file(input_file[:-3])
        output = solve(tasks, best_output)
        write_output_file(output_path, output)