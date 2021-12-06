# BENCHMARK: Greedy algorithm by deadline

from parse import read_input_file, write_output_file, read_best_output_file
from score import compute_score
import numpy as np
import os

def swap_score(tasks, order, i, j):
    if i < j:
        new_order = order[:i]
        # new_order[idx1], new_order[idx2] = new_order[idx2], new_order[idx1]
        new_order = np.append(new_order, order[j])
        new_order = np.append(new_order, order[i:j])
        new_order = np.append(new_order, order[j+1:])
    elif i > j:
        new_order = order[:j]
        # new_order[idx1], new_order[idx2] = new_order[idx2], new_order[idx1]
        new_order = np.append(new_order, order[j+1:i+1])
        new_order = np.append(new_order, order[j])
        new_order = np.append(new_order, order[i+1:])
    else:
        new_order = order
    return new_order, compute_score(tasks, new_order)

def find_best_swap(tasks, order):
    best_score = compute_score(tasks, order)
    best_order = np.array([])
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            swap_order, score = swap_score(tasks, order, i, j)
            if score > best_score:
                best_score = score
                best_order = swap_order
    return best_order

def solve(tasks, order):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing  
    """
    MAX_ITERATIONS = 100

    print('Iteration {}: {}'.format(0, compute_score(tasks, order)))
    for it in range(MAX_ITERATIONS):
        best_swap = find_best_swap(tasks, order)
        if len(best_swap) == 0:
            break
        order = best_swap
        print('Iteration {}: {}'.format(it + 1, compute_score(tasks, order)))

    t = 0
    output = []
    for i in order:
        task = tasks[i - 1]
        if t + task.get_duration() > 1440:
            continue
        output.append(task.get_task_id())
        t += task.get_duration()
    return output

# Here's an example of how to run your solver.
solver_name = "basic_genetic"
if __name__ == '__main__':
    if not os.path.exists('all_outputs/{}'.format(solver_name)):
        os.mkdir('all_outputs/{}'.format(solver_name))
        os.mkdir('all_outputs/{}/small'.format(solver_name))
        os.mkdir('all_outputs/{}/medium'.format(solver_name))
        os.mkdir('all_outputs/{}/large'.format(solver_name))

    # FOR SOLVING ALL INPUTS
    for size in os.listdir('inputs/'):
        if size not in ['small', 'medium', 'large']:
            continue
        for input_file in os.listdir('inputs/{}/'.format(size)):
            if size not in input_file:
                continue
            if size != 'large':
                continue
            input_path = 'inputs/{}/{}'.format(size, input_file)
            output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, size, input_file[:-3])
            print(input_path, output_path)
            tasks = read_input_file(input_path)
            best_output = np.array(read_best_output_file(input_file[:-3]), dtype="int")
            for task in tasks:
                if task.get_task_id() not in best_output:
                    best_output = np.append(best_output, task.get_task_id())
            output = solve(tasks, best_output)
            # write_output_file(output_path, output)