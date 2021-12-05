# BENCHMARK: Greedy algorithm always choosing locally optimal task

from parse import read_input_file, write_output_file
import os

def solve(tasks):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing  
    """
    t = 0
    order = []
    while len(tasks) > 0:
        best_task = max(tasks, key=lambda task: task.get_late_benefit(t + task.get_duration() - task.get_deadline()))
        if t + best_task.get_duration() <= 1440:
            order.append(best_task.get_task_id())
            t += best_task.get_duration()
        tasks.remove(best_task)
    return order


# Here's an example of how to run your solver.
solver_name = os.path.basename(__file__)[:-3]
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
            write_output_file(output_path, output)