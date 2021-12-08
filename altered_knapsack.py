# DP

from parse import read_input_file, write_output_file, read_best_output_file
from score import compute_score
import numpy as np
import os, time
import matplotlib.pyplot as plt

coefficients = []
profits = []

def heuristic_gen():
    a = np.random.uniform(0, 1)
    b = np.random.uniform(0, 1)
    c = np.random.uniform(-1, 1)
    # return lambda task: task.get_deadline() + a * task.get_max_benefit()
    return lambda task: task.get_deadline() / 1440 - b * task.get_max_benefit() / 100 - c * task.get_duration() / 100

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
        heuristic = heuristic_gen()
        task_ids.sort(key=lambda i: heuristic(tasks[i - 1]))
    else:
        task_ids = order
    return knapsack(tasks, task_ids)


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
    #         print(compute_score(tasks, output))
    #         write_output_file(output_path, output)

    # input_file = "medium-1.in"
    # input_size = input_file.split('-')[0]
    # input_path = 'inputs/{}/{}'.format(input_size, input_file)
    # output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, input_size, input_file[:-3])
    # tasks = read_input_file(input_path)
    # output = solve(tasks)
    # print(output, compute_score(tasks, output))

    # FOR SOLVING ALL INPUTS
    improvements = 0
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
            for task in tasks:
                if task.get_task_id() not in best_output:
                    best_output.append(task.get_task_id())
            best_score = compute_score(tasks, best_output)
            for i in range(80):
                output = solve(tasks)
                output_score = compute_score(tasks, output)
                profits.append(output_score)
                if output_score > best_score:
                    best_output = output
                    best_score = output_score
                    improvements += 1
                print("Previous Best:", compute_score(tasks, best_output), "New Best:", output_score, "Improvements:", improvements)
            write_output_file(output_path, best_output)
            
    
    # input_file = "small-56.in"
    # input_size = input_file.split('-')[0]
    # input_path = 'inputs/{}/{}'.format(input_size, input_file)
    # output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, input_size, input_file[:-3])
    # tasks = read_input_file(input_path)
    # best_output = read_best_output_file(input_file[:-3])
    # for task in tasks:
    #     if task.get_task_id() not in best_output:
    #         best_output.append(task.get_task_id())
    # best_score = compute_score(tasks, best_output)
    # for i in range(50):
    #     output = solve(tasks)
    #     output_score = compute_score(tasks, output)
    #     profits.append(output_score)
    #     if output_score > best_score:
    #         best_output = output
    #         best_score = output_score
    #     print("Previous Best:", compute_score(tasks, best_output), "New Best:", output_score)
    # write_output_file(output_path, best_output)

    # plt.figure()
    # plt.plot(coefficients, profits)
    # plt.savefig('coefficients.png')