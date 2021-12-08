# BENCHMARK: Basic genetic algorithm

from parse import read_input_file, write_output_file, read_best_output_file
from score import compute_score
import numpy as np
import os, sys

config = {
    "NUM_ITERATIONS": 1000,
    "POPULATION_SIZE": 100,
    "CROSSOVER_SIZE": 25,
    "MUTATION_RATE": 0.8,
    "THRESHOLD_FITNESS": 100000,
    "ELITISM": 2
}

# class Genome:
#     def __init__(self, population, order=np.array([], dtype="int")):
#         self.population = population
#         self.order = order
#         self.fitness = None
    
#     def calculate_fitness(self):
#         if self.fitness == None:
#             self.fitness = compute_score(self.population.tasks, self.order)
#         return self.fitness

#     def mutate(self):
#         idx1, idx2 = np.random.randint(len(self.order)), np.random.randint(len(self.order))
#         self.order[idx1], self.order[idx2] = self.order[idx2], self.order[idx1]

#     @staticmethod
#     def random_genome(population):
#         new_genome = Genome(population)
#         new_genome.order = np.random.permutation(np.arange(1, len(population.tasks) + 1))
#         return new_genome

#     @staticmethod
#     def crossover(g1, g2, next_population):
#         split_point1, split_point2 = np.random.randint(len(g1.order)), np.random.randint(len(g1.order))
#         left_split, right_split = min(split_point1, split_point2), max(split_point1, split_point2)
#         g1_cross = g1.order[left_split:right_split]
#         g2_cross = np.array([], dtype='int')
#         g1_cross_set = set(g1_cross)
#         for id in g2.order:
#             if id not in g1_cross_set:
#                 g2_cross = np.append(g2_cross, id)
#         new_order = np.concatenate((g2_cross[:left_split], g1_cross, g2_cross[left_split:]), axis=None)
#         return Genome(next_population, new_order)

class Population:
    def __init__(self, tasks, size, genomes=np.array([], dtype="int"), number=0):
        self.tasks = tasks
        self.population_size = size
        self.number = number
        self.genomes = genomes
        self.fitnesses = np.array([], dtype="int")

    @staticmethod
    def random_population(tasks, population_size):
        population = Population(tasks, population_size)
        genomes = np.empty((0, len(tasks)), int)
        for _ in range(population_size):
            genome = np.random.permutation(np.arange(1, len(population.tasks) + 1))
            genomes = np.append(genomes, np.array([genome]), axis=0)
        population.genomes = genomes
        population.fitnesses = population.calculate_population_fitness()
        return population

    @staticmethod
    def population_from_initial_order(tasks, population_size, order):
        population = Population.random_population(tasks, population_size)
        population.genomes[0,:] = order
        population.fitnesses = population.calculate_population_fitness()
        return population
    
    def calculate_genome_fitness(self, genome):
        return compute_score(self.tasks, genome)

    def calculate_population_fitness(self):
        fitness_array = np.array([self.calculate_genome_fitness(genome) for genome in self.genomes])
        return fitness_array

    def best_genome(self, k=config["ELITISM"]):
        if k == 1:
            return np.array([self.genomes[np.argmax(self.fitnesses)]])
        ind = np.argpartition(self.fitnesses, -k)[-k:]
        return self.genomes[ind]

    def selection(self, k=2):
        candidates = [np.random.randint(0, self.population_size) for i in range(k)]
        winner = max(candidates, key=lambda i: self.fitnesses[i])
        return self.genomes[winner]

    def elite(self, num_parents):
        top_genome_ids = np.argpartition(self.fitnesses, -num_parents)[-num_parents:]
        return [self.genomes[i] for i in top_genome_ids]

    def crossover(self, genome1, genome2):
        new_genome = np.array([], dtype="int")
        split_point1, split_point2 = np.random.randint(0, len(genome1)), np.random.randint(0, len(genome1))
        left_split, right_split = min(split_point1, split_point2), max(split_point1, split_point2)
        g1_cross = genome1[left_split:right_split]
        g2_cross = genome2[~np.in1d(genome2, g1_cross)]
        new_genome = np.append(new_genome, g2_cross[:left_split])
        new_genome = np.append(new_genome, g1_cross)
        new_genome = np.append(new_genome, g2_cross[left_split:])
        # np.concatenate((g2_cross[:left_split], g1_cross, g2_cross[left_split:]), axis=None)
        return new_genome

    def mutate(self, genome):
        idx1, idx2 = np.random.randint(len(genome)), np.random.randint(len(genome))
        new_genome = np.copy(genome)
        new_genome[idx1], new_genome[idx2] = new_genome[idx2], new_genome[idx1]
        return new_genome

    def stagnate(self):
        for i in range(self.population_size // 2):
            genome = np.random.permutation(np.arange(1, len(self.tasks) + 1))
            self.genomes[self.population_size - i - 1] = genome

    def next_population(self, mutation_rate):
        next_pop = Population(self.tasks, self.population_size, np.array([], dtype="int"), self.number + 1)
        next_pop_genomes = np.empty((0, len(self.tasks)), int)
        if config["ELITISM"]: next_pop_genomes = np.append(next_pop_genomes, self.best_genome(), axis=0)
        for i in range(config["ELITISM"], self.population_size):
            parent1, parent2 = self.selection(), self.selection()
            offspring = self.crossover(parent1, parent2)
            if np.random.random() <= mutation_rate:
                offspring = self.mutate(offspring)
            next_pop_genomes = np.append(next_pop_genomes, np.array([offspring]), axis=0)
        next_pop.genomes = next_pop_genomes
        next_pop.fitnesses = next_pop.calculate_population_fitness()
        return next_pop

    def __str__(self):
        # return 'Population {}: Fitness - {}, Best Genome - {}'.format(self.number, np.max(self.fitnesses), self.best_genome())
        return 'Population {}: Fitness - {}'.format(self.number, np.max(self.fitnesses))


def solve(tasks, initial_genome=np.array([], dtype="int")):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing  
    """
    if len(initial_genome) == 0:
        population = Population.random_population(tasks, config["POPULATION_SIZE"])
    else:
        population = Population.population_from_initial_order(tasks, config["POPULATION_SIZE"], initial_genome)
    print(population)
    print(population.genomes)
    prev_fitness = np.max(population.fitnesses)
    stagnation_count = 0
    for i in range(config["NUM_ITERATIONS"]):
        population = population.next_population(config["MUTATION_RATE"])
        best_fitness = np.max(population.fitnesses)
        if prev_fitness == best_fitness:
            stagnation_count += 1
        prev_fitness = best_fitness
        if (i + 1) % 100 == 0:
            print(population)
            print(population.genomes)
        if best_fitness >= config["THRESHOLD_FITNESS"]:
            break
    best_genome = population.best_genome(1)[0]

    t = 0
    output = []
    for i in best_genome:
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

    # # FOR SOLVING ALL INPUTS
    # improvements = 0
    # for size in os.listdir('inputs/'):
    #     if size not in ['small', 'medium', 'large']:
    #         continue
    #     for input_file in os.listdir('inputs/{}/'.format(size)):
    #         if size not in input_file:
    #             continue
    #         if size != 'small':
    #             continue
    #         input_path = 'inputs/{}/{}'.format(size, input_file)
    #         output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, size, input_file[:-3])
    #         print(input_path, output_path)
    #         tasks = read_input_file(input_path)
    #         best_output = np.array(read_best_output_file(input_file[:-3]), dtype="int")
    #         for task in tasks:
    #             if task.get_task_id() not in best_output:
    #                 best_output = np.append(best_output, task.get_task_id())
    #         output = solve(tasks, best_output)
    #         output_score = compute_score(tasks, output)
    #         best_output_score = compute_score(tasks, best_output)
    #         if output_score > best_output_score:
    #             improvements += 1
    #         print("Previous Best:", best_output_score, "New Best:", output_score, "Improvements:", improvements)
    #         write_output_file(output_path, output)

    # FOR SOLVING ONE INPUT
    input_file = "small-45.in"
    input_size = input_file.split('-')[0]
    input_path = 'inputs/{}/{}'.format(input_size, input_file)
    output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, input_size, input_file[:-3])
    tasks = read_input_file(input_path)
    best_output = np.array(read_best_output_file(input_file[:-3]), dtype="int")
    for task in tasks:
        if task.get_task_id() not in best_output:
            best_output = np.append(best_output, task.get_task_id())
    output = solve(tasks, best_output)
    write_output_file(output_path, output)

