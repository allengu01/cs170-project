# LINEAR PROGRAMMING
from parse import read_input_file, write_output_file
from score import compute_score
from mip import *
import mip
import os

def add_linear_approx(model, p, t, deadline, max_benefit):
    c = 0.2
    k = c * max_benefit * -0.0170
    model += p <= k * t + (max_benefit - k * deadline)

def solve(tasks):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing  
    """
    n = len(tasks)
    m = Model(sense=MAXIMIZE)
    b = [[m.add_var(var_type=BINARY) for _ in range(n)] for _ in range(n)]
    p = [m.add_var(var_type=CONTINUOUS, lb=0, ub=100) for _ in range(n)]
    t = [m.add_var(var_type=INTEGER, lb=0, ub=1440) for _ in range(n)]
    # d = [m.add_var(var_type=BINARY) for _ in range(n)]
    for i in range(n):
        task = tasks[i]
        m += p[i] <= task.get_max_benefit()
        add_linear_approx(m, p[i], t[i], task.get_deadline(), task.get_max_benefit())
        m += t[i] == xsum(b[j][i] for j in range(0, n) if i != j)
        # m += t[i] <= 1440 + 100000 * (1 - d[i])

    
    for i in range(n):
        for j in range(i + 1, n):
            m += b[i][j] == b[j][i]

    m.objective = xsum(p[i] for i in range(n))
    m.optimize()
    order_zipped = [(t[i], i) for i in range(n)]
    order_zipped.sort()
    order = [pair[1] for pair in order_zipped]

    t = 0
    output = []
    for i in order:
        task = tasks[i - 1]
        if t + task.get_duration() > 1440:
            continue
        output.append(task.get_task_id())
        t += task.get_duration()
    print(output, compute_score(tasks, output))


# Here's an example of how to run your solver.
solver_name = os.path.basename(__file__)[:-3]
if __name__ == '__main__':
    if not os.path.exists('all_outputs/{}'.format(solver_name)):
        os.mkdir('all_outputs/{}'.format(solver_name))
        os.mkdir('all_outputs/{}/small'.format(solver_name))
        os.mkdir('all_outputs/{}/medium'.format(solver_name))
        os.mkdir('all_outputs/{}/large'.format(solver_name))

    # FOR SOLVING ALL INPUTS
    # for size in os.listdir('inputs/'):
    #     if size not in ['small', 'medium', 'large']:
    #         continue
    #     for input_file in os.listdir('inputs/{}/'.format(size)):
    #         if size not in input_file:
    #             continue
    #         if size != 'medium':
    #             continue
    #         input_path = 'inputs/{}/{}'.format(size, input_file)
    #         output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, size, input_file[:-3])
    #         print(input_path, output_path)
    #         tasks = read_input_file(input_path)
    #         output = solve(tasks)
    #         write_output_file(output_path, output)

    # FOR SOLVING ONE INPUT
    input_file = "medium-1.in"
    input_size = input_file.split('-')[0]
    input_path = 'inputs/{}/{}'.format(input_size, input_file)
    output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, input_size, input_file[:-3])
    tasks = read_input_file(input_path)
    output = solve(tasks)
    # write_output_file(output_path, output)