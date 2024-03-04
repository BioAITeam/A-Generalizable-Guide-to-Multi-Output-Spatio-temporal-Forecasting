import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from darts.metrics import r2_score, mse
from Model import *


def mutation(individual, parameters_bounds, mutation_rate, exp = []):
    """
    Mutate an individual by randomly changing some of its parameters.

    Parameters:
    - individual: a list of parameters
    - parameters_bounds: a list of tuples that define the lower and upper bounds for each parameter
    - mutation_rate: the probability of each parameter being mutated

    Returns:
    - mutated_individual: the mutated individual
    """
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
          if i in exp:
            parameter_min, _ = parameters_bounds[i]
            parameter_max = mutated_individual[i-1]-1
          else:
            parameter_min, parameter_max = parameters_bounds[i]
          if isinstance(parameter_min, int):
              mutated_individual[i] = random.randint(parameter_min, parameter_max)
          else:
              mutated_individual[i] = random.uniform(parameter_min, parameter_max)
    return mutated_individual

def crossover(parent1, parent2):
    """
    Create a new individual by randomly combining the parameters of two parents.

    Parameters:
    - parent1: a list of parameters
    - parent2: a list of parameters

    Returns:
    - child: the new individual created by crossover
    """
    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child



def initialize_population(pop_size, parameters_bounds,exp = []):
    """
    Create an initial population of random individuals.

    Parameters:
    - pop_size: the size of the population
    - parameters_bounds: a list of tuples that define the lower and upper bounds for each parameter

    Returns:
    - population: a list of individuals
    """
    population = []
    for i in range(pop_size):
        individual = []
        for j in range(len(parameters_bounds)):
            if j in exp:
              parameter_min, _ = parameters_bounds[j]
              parameter_max = individual[-1]-1
            else:
              parameter_min, parameter_max = parameters_bounds[j]
            if isinstance(parameter_min, int):
                individual.append(random.randint(parameter_min, parameter_max))
            else:
                individual.append(random.uniform(parameter_min, parameter_max))
        population.append(individual)
    return population

def Genetic_algoritm (train_set, test_set, recursive, Generations, top, cross, init_pop, model_name, epoch):
  parameters_bounds = choose_parameters(model_name)
  population_param = initialize_population(init_pop, parameters_bounds)
  scores = []
  gen = 1
  for param in tqdm(population_param):
    model = Model_selection(model_name, param, recursive, epoch)
    model.fit(train_set, verbose = False)
    forecast = model.historical_forecasts(test_set,start=test_set.get_timestamp_at_point(param[0]),forecast_horizon=param[1],retrain=False,verbose=False)
    scores.append(r2_score(test_set.map(lambda x:x+1), forecast.map(lambda x:x+1)))
  best_individual = population_param[scores.index(max(scores))]
  best_fitness = max(scores)
  print(f"Generation 1: Model = {model_name}, Best individual = {best_individual}, Fitness = {best_fitness}")
  Best = np.array([[gen]+best_individual+[best_fitness]])
  for i in range(2,Generations+1):
    selected_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top]
    parents = [population_param[i] for i in selected_indices]
    new_population = 0
    pop_index =  len(population_param)
    while new_population < cross:
      # Choose two parents randomly
      parent1, parent2 = random.choices(parents, k=2)
      # Perform crossover and mutation to create a new individual
      child = crossover(parent1, parent2)
      child = mutation(child, parameters_bounds, 0.1)
      population_param.append(child)
      new_population +=1
    
    for param in tqdm(population_param[pop_index:]):
      model = Model_selection(model_name, param, recursive = False)
      model.fit(train_set, verbose = False)
      forecast = model.historical_forecasts(test_set,start=test_set.get_timestamp_at_point(param[0]),forecast_horizon=param[1],retrain=False,verbose=False)
      scores.append(r2_score(test_set.map(lambda x:x+1), forecast.map(lambda x:x+1)))
    
    # Print the best individual and its fitness score for each generation
    best_individual = population_param[scores.index(max(scores))]
    best_fitness = max(scores)
    print(f"Generation {i}: Model = {model_name}, Best individual = {best_individual}, Fitness = {best_fitness}")
    Best2 = np.array([[int(i)]+best_individual+[best_fitness]])
    Best = np.append(Best, Best2, axis=0)

  return Best