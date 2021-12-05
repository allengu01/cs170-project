# BENCHMARK: Basic genetic algorithm

from parse import read_input_file, write_output_file, read_best_output_file
from score import compute_score
import numpy as np
import os

config = {
    "NUM_ITERATIONS": 1000,
    "POPULATION_SIZE": 30,
    "CROSSOVER_SIZE": 25,
    "MUTATION_RATE": 0.8,
}

class Genome:
    def __init__(self, population, order=[]):
        self.population = population
        self.order = order
        self.fitness = None
    
    def calculate_fitness(self):
        if self.fitness == None:
            self.fitness = compute_score(self.population.tasks, self.order)
        return self.fitness

    def mutate(self):
        idx1, idx2 = np.random.randint(len(self.order)), np.random.randint(len(self.order))
        self.order[idx1], self.order[idx2] = self.order[idx2], self.order[idx1]

    @staticmethod
    def random_genome(population):
        new_genome = Genome(population)
        new_genome.order = np.random.permutation(np.arange(1, len(population.tasks) + 1))
        return new_genome

    @staticmethod
    def crossover(g1, g2, next_population):
        split_point1, split_point2 = np.random.randint(len(g1.order)), np.random.randint(len(g1.order))
        left_split, right_split = min(split_point1, split_point2), max(split_point1, split_point2)
        g1_cross = g1.order[left_split:right_split]
        g2_cross = np.array([], dtype='int')
        g1_cross_set = set(g1_cross)
        for id in g2.order:
            if id not in g1_cross_set:
                g2_cross = np.append(g2_cross, id)
        new_order = np.concatenate((g2_cross[:left_split], g1_cross, g2_cross[left_split:]), axis=None)
        return Genome(next_population, new_order)

class Population:
    def __init__(self, tasks, size, genomes=[], number=0):
        self.tasks = tasks
        self.population_size = size
        self.number = number
        self.genomes = genomes

    @staticmethod
    def random_population(tasks, population_size):
        population = Population(tasks, population_size)
        for _ in range(population_size):
            genome = Genome.random_genome(population)
            population.genomes.append(genome)
        return population

    @staticmethod
    def population_from_initial_order(tasks, population_size, order):
        pop = Population(tasks, population_size)
        initial_genome = Genome(pop, order)
        return Population(tasks, population_size, [initial_genome for _ in range(population_size)])
            
    def calculate_population_fitness(self):
        fitness_array = np.array([genome.calculate_fitness() for genome in self.genomes])
        return fitness_array

    def best_genome(self):
        fitness_array = self.calculate_population_fitness()
        return self.genomes[np.argmax(fitness_array)]

    def select_parents(self, fitness, num_parents):
        top_genome_ids = np.argpartition(fitness, -num_parents)[-num_parents:]
        return [self.genomes[i] for i in top_genome_ids]

    def crossover(self, parents, offspring_size, next_population):
        offspring = []
        for i in range(offspring_size):
            parent1, parent2 = parents[i % len(parents)], parents[(i + 1) % len(parents)]
            o = Genome.crossover(parent1, parent2, next_population)
            offspring.append(o)
        return offspring

    def mutate(self, crossover_offspring, mutation_rate):
        for genome in crossover_offspring:
            if np.random.random() <= mutation_rate: # mutate
                genome.mutate()
        return crossover_offspring

    def next_population(self, crossover_size, mutation_rate):
        next_pop = Population(self.tasks, self.population_size, [], self.number + 1)
        fitness_array = self.calculate_population_fitness()
        parents = self.select_parents(fitness_array, self.population_size - crossover_size)
        offspring = self.crossover(parents, crossover_size, next_pop)
        mutated_offspring = self.mutate(offspring, mutation_rate)
        next_genomes = parents + mutated_offspring
        next_pop.genomes = next_genomes
        return next_pop

    def __str__(self):
        return 'Population {}: Fitness - {}'.format(self.number, self.best_genome().fitness)
    

def solve(tasks, initial_genome=[]):
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
    for i in range(config["NUM_ITERATIONS"]):
        population = population.next_population(config["CROSSOVER_SIZE"], config["MUTATION_RATE"])
        if (i + 1) % 100 == 0:
            print(population)
    best_genome = population.best_genome()

    t = 0
    output = []
    for i in best_genome.order:
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
            input_path = 'inputs/{}/{}'.format(size, input_file)
            output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, size, input_file[:-3])
            print(input_path, output_path)
            tasks = read_input_file(input_path)
            best_output = np.array(read_best_output_file(input_file[:-3]), dtype="int")
            for task in tasks:
                if task.get_task_id() not in best_output:
                    best_output = np.append(best_output, task.get_task_id())
            output = solve(tasks, best_output)
            write_output_file(output_path, output)

    # FOR SOLVING ONE INPUT
    # input_file = "small-1"
    # input_size = input_file.split('-')[0]
    # input_path = 'inputs/{}/{}.in'.format(input_size, input_file)
    # output_path = 'all_outputs/{}/{}/{}.out'.format(solver_name, input_size, input_file[:-3])
    # tasks = read_input_file(input_path)
    # best_output = read_best_output_file(input_file)
    # for task in tasks:
    #     if task.get_task_id() not in best_output:
    #         best_output.append(task.get_task_id())
    # output = solve(tasks, best_output)
    # write_output_file(output_path, output)

