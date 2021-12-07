# BENCHMARK: Greedy algorithm by deadline

from parse import read_input_file, write_output_file, read_best_output_file
from score import compute_score
import os, numpy as np

def swap_score(tasks, order, i, j):
    order[i], order[j] = order[j], order[i]
    score = compute_score(tasks, order)
    order[i], order[j] = order[j], order[i]
    # if i < j:
    #     new_order = order[:i] + [order[j]] + order[i:j] + order[j+1:]
    #     # new_order[idx1], new_order[idx2] = new_order[idx2], new_order[idx1]
    # elif i > j:
    #     new_order = order[:j] + order[:j] + order[j+1:i+1] + [order[j]] + order[i+1:]
    #     # new_order[idx1], new_order[idx2] = new_order[idx2], new_order[idx1]
    # else:
    #     new_order = order
    # return compute_score(tasks, new_order)
    return score

def find_best_swap(tasks, order):
    best_score = compute_score(tasks, order)
    best_swap = None
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            score = swap_score(tasks, order, i, j)
            if score > best_score:
                best_score = score
                best_swap = (i, j)
    return best_swap

def solve(tasks, order):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing  
    """
    MAX_ITERATIONS = 10

    for it in range(MAX_ITERATIONS):
        best_swap = find_best_swap(tasks, order)
        if best_swap == None:
            break
        i, j = best_swap
        order[i], order[j] = order[j], order[i]
        if (it + 1) % 1 == 0:
            print('Iteration {}: {}'.format(it + 1, compute_score(tasks, order)))
        print(type(order[0]))
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

    # FOR SOLVING ONE INPUT
    input_file = "medium-3.in"
    input_size = input_file.split('-')[0]
    input_path = 'inputs/{}/{}'.format(input_size, input_file)
    output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, input_size, input_file[:-3])
    tasks = read_input_file(input_path)
    best_output = read_best_output_file(input_file[:-3])
    for task in tasks:
        if task.get_task_id() not in best_output:
            best_output.append(task.get_task_id())
    output = solve(tasks, best_output)
    write_output_file(output_path, output)