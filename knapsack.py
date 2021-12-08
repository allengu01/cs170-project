# DP

from parse import read_input_file, write_output_file, read_best_output_file
from score import compute_score
import numpy as np
import os, time

heuristic_map = {
    "deadline": lambda task: task.get_deadline(),
    "benefit": lambda task: -task.get_benefit(),
    "ratio": lambda task: -task.get_benefit()/task.get_deadline(),
    "deadline_benefit": lambda task: task.get_deadline() / 1440 - task.get_max_benefit() / 1000,
    "deadline_duration_plain": lambda task: task.get_deadline() - task.get_duration(),
    "deadline_duration": lambda task: task.get_deadline() - task.get_duration() if task.get_deadline() - task.get_duration() >= 0 else 100000,
    "custom1": lambda task: (task.get_deadline() - task.get_duration()) / 1440 - task.get_late_benefit(task.get_duration() - task.get_deadline()) / 1000,
    "benefit_duration": lambda task: -task.get_max_benefit() + task.get_duration(),
    "duration": lambda task: task.get_duration(),
    "more_benefit_deadline": lambda task: task.get_deadline() / 1440 - 1.70 * task.get_max_benefit() / 100
}
heuristic_key = "more_benefit_deadline"

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

def solve(tasks, order=[]):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing  
    """
    if len(order) == 0:
        task_ids = [task.get_task_id() for task in tasks]
        task_ids.sort(key=lambda i: heuristic(tasks, i, heuristic_map[heuristic_key]))
    else:
        task_ids = order
    return knapsack(tasks, task_ids)


# Here's an example of how to run your solver.
improvements = 0
# solver_name = '{}-{}'.format(os.path.basename(__file__)[:-3], heuristic_key)
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
    #         best_output = read_best_output_file(input_file[:-3])
    #         for task in tasks:
    #             if task.get_task_id() not in best_output:
    #                 best_output.append(task.get_task_id())
    #         output_score = compute_score(tasks, output)
    #         best_output_score = compute_score(tasks, best_output)
    #         if output_score > best_output_score:
    #             improvements += 1
    #         print("Previous Best:", best_output_score, "New Best:", output_score, "Improvements:", improvements)
    #         write_output_file(output_path, output)

    # input_file = "medium-1.in"
    # input_size = input_file.split('-')[0]
    # input_path = 'inputs/{}/{}'.format(input_size, input_file)
    # output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, input_size, input_file[:-3])
    # tasks = read_input_file(input_path)
    # output = solve(tasks)
    # print(output, compute_score(tasks, output))

    # FOR SOLVING ALL INPUTS WITH PREVIOUS BEST
    solver_name = os.path.basename(__file__)[:-3]
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
            best_output = read_best_output_file(input_file[:-3])
            best_output_full = best_output.copy()
            for task in tasks:
                if task.get_task_id() not in best_output:
                    best_output_full.append(task.get_task_id())
            output = solve(tasks, best_output_full)
            output_score = compute_score(tasks, output)
            best_output_score = compute_score(tasks, best_output)
            if output_score > best_output_score:
                improvements += 1
                best_output = output
            print("Previous Best:", best_output_score, "New Best:", output_score, "Improvements:", improvements)
            write_output_file(output_path, best_output)