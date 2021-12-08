# BENCHMARK: Greedy algorithm always choosing locally optimal task

from parse import read_input_file, write_output_file
from score import compute_score
import numpy as np
import os
import time

from score import compute_score

def solve(tasks):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing  
    """
    def heuristic(task, t):
        return  task.get_late_benefit(t + task.get_duration() - task.get_deadline()) / task.get_duration()
        # return task.get_late_benefit(t + task.get_duration() - task.get_deadline() - (task.get_deadline() - t - task.get_duration()))

    starttime = time.time()
    t = 0
    order = []
    while len(tasks) > 0:
        best_task = max(tasks, key=lambda task: heuristic(task, t))
        if t + best_task.get_duration() <= 1440:
            order.append(best_task.get_task_id())
            t += best_task.get_duration()
        tasks.remove(best_task)
    endtime = time.time()
    print(endtime - starttime)
    return order


# Here's an example of how to run your solver.
solver_name = os.path.basename(__file__)[:-3]
if __name__ == '__main__':
    if not os.path.exists('all_outputs/{}'.format(solver_name)):
        os.mkdir('all_outputs/{}'.format(solver_name))
        os.mkdir('all_outputs/{}/small'.format(solver_name))
        os.mkdir('all_outputs/{}/medium'.format(solver_name))
        os.mkdir('all_outputs/{}/large'.format(solver_name))
    # for size in os.listdir('inputs/'):
    #     if size not in ['small', 'medium', 'large']:
    #         continue
    #     for input_file in os.listdir('inputs/{}/'.format(size)):
    #         if size not in input_file:
    #             continue
    #         input_path = 'inputs/{}/{}'.format(size, input_file)
    #         output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, size, input_file[:-3])
    #         print(input_path, output_path)
    #         tasks = read_input_file(input_path)
    #         output = solve(tasks)
    #         write_output_file(output_path, output)

    input_file = "large-160.in"
    input_size = input_file.split('-')[0]
    input_path = 'inputs/{}/{}'.format(input_size, input_file)
    output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, input_size, input_file[:-3])
    tasks = read_input_file(input_path)
    tasks_copy = tasks.copy()
    
    output = solve(tasks)
    print(output, compute_score(tasks_copy, output))