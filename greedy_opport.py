# Utilizing an opportunity cost metric in order to select the next igloo

from parse import read_input_file, write_output_file
import os

def solve(tasks):
    
    t = 0
    time_passed = 0
    order = []
    while len(tasks) > 0:
        opportunity_cost = []
        for i in range(len(tasks)):
            opportunity_cost.append(calc_opportunity_cost(tasks, i, time_passed))
        min_index = opportunity_cost.index(min(opportunity_cost))
        time_passed += tasks[min_index].get_duration()
        order.append(min_index + 1)
        tasks.pop(min_index)
    return order

def calc_opportunity_cost(tasks, i, time_passed):
    opport_cost = 0
    for task in tasks:
        if tasks[i] == task:
            opport_cost -= task.get_late_benefit(task.get_deadline() - time_passed)
        else:
            opport_cost += task.get_late_benefit(task.get_deadline() - time_passed)
    return opport_cost

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

