from parse import read_input_file, read_output_file, read_best_output_file
from leaderboard import save_scores
import os

def compute_score(tasks, order):
    t = 0
    score = 0
    for task_id in order:
        task = tasks[task_id - 1]
        if t + task.get_duration() <= 1440:
            t += task.get_duration()
            score += task.get_late_benefit(t - task.get_deadline())
        # else:
        #     print(task)
    return score

# Here's an example of how to run your solver.
solver_name = "random_swapping"
if __name__ == '__main__':
    scores = {}
    for size in os.listdir('inputs/'):
        if size not in ['small', 'medium', 'large']:
            continue
        for input_file in os.listdir('inputs/{}/'.format(size)):
            if size not in input_file:
                continue
            input_path = 'inputs/{}/{}'.format(size, input_file)
            output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, size, input_file[:-3])
            if not os.path.exists(output_path):
                continue
            print(input_path, output_path)
            tasks = read_input_file(input_path)
            order = read_output_file(output_path)
            scores[input_file[:-3]] = compute_score(tasks, order)
    save_scores(solver_name, scores)

    # input_file = "large-145.in"
    # input_size = input_file.split('-')[0]
    # input_path = 'inputs/{}/{}'.format(input_size, input_file)
    # output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, input_size, input_file[:-3])
    # tasks = read_input_file(input_path)
    # best_output = read_best_output_file(input_file[:-3])
    # for task in tasks:
    #     if task.get_task_id() not in best_output:
    #         best_output.append(task.get_task_id())
    # print(best_output)
    # best_score = compute_score(tasks, best_output)
    # print(best_score)

