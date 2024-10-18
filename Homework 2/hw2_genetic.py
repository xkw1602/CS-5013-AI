import pandas as pd
import random
import math
import matplotlib.pyplot as plt
# Read data set and handle missing values
df = pd.read_csv('CreditCard.csv')
df = df.dropna(subset=['CreditApprove', 'Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID'])

# Encode binary attributes
df['Gender'] = df['Gender'].replace({'M':1, 'F':0})
df['CarOwner'] = df['CarOwner'].replace({'Y': 1, 'N' : 0})
df['PropertyOwner'] = df['PropertyOwner'].replace({'Y': 1, 'N': 0})

# Genetic algorithm paramteters
population_size = 6
generations = 100
mutation_rate = 0.05
CROSSOVER = 3

# function to randomly generate initial w-values
def initialize_population(size):
    return [[random.choice([1, -1]) for _ in range(6)] for _ in range(size)]

# initialize record for calculation
record = [-1, -1, -1, -1, -1, -1]

# Ignored one record due to missing value, so 339 records instead of 340
num_records = 339

# error calculation function
def calculate_error(w):
    sum_squared_errors = 0

    for index, row in df.iterrows():
        y = row['CreditApprove']
        record[0] = row['Gender']
        record[1] = row['CarOwner']
        record[2] = row['PropertyOwner']
        record[3] = row['#Children']
        record[4] = row['WorkPhone']
        record[5] = row['Email_ID']

        fx = 0
        for x in range(6):
            fx += w[x]*record[x]

        sum_squared_errors += (fx - y)**2
        
    er = sum_squared_errors/num_records
    return(er)

# fitness calculation function
def calculate_fitness(w):
    error = calculate_error(w)
    return math.e**(-1 * error)

# crossover function for offspring creation
def crossover(parent1, parent2):
    offspring = parent1[:CROSSOVER] + parent2[CROSSOVER:]
    return offspring

# parent selection function based on calculated probabilites
def select_parent(population, probabilites):
    r = random.random()
    for i, p in enumerate(probabilites):
        if r <= p:
            return i

# Mutation function
def mutate(chromosome):
    for i in range(len(chromosome)): 
        r = random.random()
        if r <= mutation_rate:
            chromosome[i] = -1 * chromosome[i]  
    return chromosome   

#--------------------------------------------------------------------------
# Genetic algorithm execution
population = initialize_population(population_size)

total_generations = 0
best_errors_per_generation = []
for generation in range(generations):
    total_generations += 1
    generation_best_error = float('inf')

    # Calculate fitness for each chromosome and find best error 
    fitness_array = []
    for chromosome in population:
        error = calculate_error(chromosome)
        if error < generation_best_error:
            generation_best_error = error
        fitness_array.append(calculate_fitness(chromosome))
    
    best_errors_per_generation.append(generation_best_error)

    # Use fitness to calculate proportional probabilities for selection
    total_fitness_values = sum(fitness_array)
    probabilities = []
    cumulative_probabilites = []
    cumulative_sum = 0
    for value in fitness_array:
        probability = value/total_fitness_values
        probabilities.append(probability)
        cumulative_sum += probability
        cumulative_probabilites.append(cumulative_sum)

    # Generate new population
    new_population = []
    while len(new_population) < population_size:
        # Select parent1
        parent1_index = select_parent(population, cumulative_probabilites)
        parent1 = population[parent1_index]
         # Select parent2
        parent2_index = parent1_index
        while(parent2_index == parent1_index):
            parent2_index = select_parent(population, cumulative_probabilites)
        parent2 = population[parent2_index]
        # Generate offspring
        offspring = crossover(parent1, parent2)
        offspring = mutate(offspring)
        new_population.append(offspring)
    
    population = new_population

best_chromosome = None
best_error = float('inf')
for chromosome in population:
    error = calculate_error(chromosome)
    if error < best_error:
        best_error = error
        best_chromosome = chromosome

print(f'Final Population: {population}')
print(f'Total Generations: {total_generations}\nBest chromosome from final population: {best_chromosome}\nError: {best_error}')

plt.plot(range(len(best_errors_per_generation)), best_errors_per_generation, marker='o')
plt.title('Generation vs Best Error')
plt.xlabel('Generation')
plt.ylabel('Best Error')
plt.grid(True)
plt.show()