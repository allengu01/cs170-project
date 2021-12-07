# DP

from parse import read_input_file, write_output_file
from score import compute_score
import numpy as np
import os, time

heuristic_map = {
    "deadline": lambda task: task.get_deadline(),
    "benefit": lambda task: -task.get_max_benefit(),
    "ratio": lambda task: -task.get_max_benefit()/task.get_deadline()
    
}
heuristic_key = "benefit"

def heuristic(tasks, i, heuristic_func):
    task = tasks[i - 1]
    return heuristic_func(task)

def knapsack(tasks, indices):
    starttime = time.time()
    dp = [[0] * (1440 + 1) for _ in range(len(tasks) + 1)]
    for i in range(1, len(tasks)):
        for j in range(1441):
            task_id = indices[i - 1]
            task = tasks[task_id - 1]
            duration = task.get_duration()
            if duration <= j:
                benefit = task.get_late_benefit(j - task.get_deadline())
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-duration] + benefit)
            else:
                dp[i][j] = dp[i-1][j]

    # print(dp)
    def reconstruct(i, j):
        if i == 0:
            return []
        task_id = indices[i - 1]
        task = tasks[task_id - 1]
        duration = task.get_duration()
        if dp[i][j] > dp[i - 1][j]:
            output = reconstruct(i - 1, j - duration)
            output.append(task_id)
            return output
        else:
            return reconstruct(i - 1, j)

    output = reconstruct(len(tasks), 1440)
    endtime = time.time()
    print(endtime - starttime)
    return output

def solve(tasks):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing  
    """
    task_ids = [task.get_task_id() for task in tasks]
    task_ids.sort(key=lambda i: heuristic(tasks, i, heuristic_map[heuristic_key]))
    # for task_id in task_ids:
    #     print(tasks[task_id - 1])
    return knapsack(tasks, task_ids)


# Here's an example of how to run your solver.
solver_name = '{}-{}'.format(os.path.basename(__file__)[:-3], heuristic_key)
if __name__ == '__main__':
    if not os.path.exists('all_outputs/{}'.format(solver_name)):
        os.mkdir('all_outputs/{}'.format(solver_name))
        os.mkdir('all_outputs/{}/small'.format(solver_name))
        os.mkdir('all_outputs/{}/medium'.format(solver_name))
        os.mkdir('all_outputs/{}/large'.format(solver_name))
    for size in os.listdir('inputs/'):
        if size not in ['small', 'medium', 'large']:
            continue
        for input_file in os.listdir('inputs/{}/'.format(size)):
            if size not in input_file:
                continue
            input_path = 'inputs/{}/{}'.format(size, input_file)
            output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, size, input_file[:-3])
            print(input_path, output_path)
            tasks = read_input_file(input_path)
            output = solve(tasks)
            print(compute_score(tasks, output))
            write_output_file(output_path, output)

    # input_file = "medium-1.in"
    # input_size = input_file.split('-')[0]
    # input_path = 'inputs/{}/{}'.format(input_size, input_file)
    # output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, input_size, input_file[:-3])
    # tasks = read_input_file(input_path)
    # output = solve(tasks)
    # print(output, compute_score(tasks, output))