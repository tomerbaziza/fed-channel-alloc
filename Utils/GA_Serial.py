
import numpy as np 
from threading import Thread, Lock
# import multiprocessing
import pickle 
import os 
import sys 
import matplotlib.pyplot as plt 
import time

def extract_number_form_name(file_name):
    d = []
    number = ''
    for cha in file_name:
        if cha in '0,1,2,3,4,5,6,7,8,9':
            number += cha
            
    return int(number)


"""With mutex 183 sec for simple function x1**2 + x2**2 + x3**2, without 150 sec"""
mutex = Lock()

            
class Genetic_algorithm():
    def __init__(self, initialization_function = None, options = {}):

        self.initialization = initialization_function
        
        self.options = options
        
        self.options['technique'] = self.options.get('technique','RoulleteWheel')
        
        self.best_chromosome = None
    
    def create_chromosome(self):
        chromosome = []
        for i in range(self.number_of_genes):
            type_i = self.parameters[i]
            
            if type_i == "discrete":
                gene = np.random.choice(self.search_range[i])
            
            else:
                # continuse
                minimum, maximum = self.search_range[i]
                
                gene = np.random.random() * (maximum - minimum) + minimum
                
            chromosome.append(gene)
            
        return chromosome
    
    def create_population(self, population_size):
        population = []
        for p in range(population_size):
            chromosome = self.create_chromosome()
            population.append(chromosome)
        
        return population
    
    def create_mutation(self,population, eps = 0.2, std = 1.0):
        index= np.random.choice(self.population_size)
        chromosome = list(population[index])
        
        for i in range(self.number_of_genes):
            typ_i = self.parameters[i]
            domain = self.search_range[i]
            
            if typ_i == "discrete":
                if np.random.random()<eps:
                    chromosome[i] = np.random.choice(domain)
                else:
                    continue
            else: # for continues params 
                    
                min_value,max_value = domain
                mean = chromosome[i]
                mutant_value = np.random.normal(scale = std) + mean 
                
                mutant_value = min(max_value, mutant_value)
                mutant_value = max(min_value, mutant_value)
                chromosome[i] = mutant_value
        
        return chromosome
                  
        
    def update_population(self, population, vals,ids,eps = 0.2,std = 1.0):
        
        
        ### mutation 20%
        ## crossOver_60%
        ## Elits 20%
        
        amount_of_elits = int(self.population_size * self.elits)
        amount_of_mutation = int(self.population_size * self.mutation)
        amount_of_crossover = self.population_size - amount_of_elits - amount_of_mutation
        
        if amount_of_elits == 0:
            amount_of_elits = 1
            amount_of_mutation = 1
            amount_of_crossover = self.population_size - 2
        # orgenized from small to big 

        vals, population,ids = zip(*sorted(zip(vals,population,ids), key = lambda x: x[0]))
        vals = list(vals)
        population = list(population)
        ids = list(ids)
        old_pop = population.copy()
        #Update best result
        
        if self.operation == 'maximization':
            if vals[-1] > self.best_result:
                self.best_result = vals[-1]
                self.best_chromosome = population[-1]
                print("The new best result:", self.best_result)
                print("The chromosom:", self.best_chromosome)
                
        else:
            if vals[0] < self.best_result:
                self.best_result = vals[0]
                self.best_chromosome = population[0]
                print("The new best result:", self.best_result)
                print("The chromosom:", self.best_chromosome)

        self.all_results.append(self.best_result)
        print("all_res:", self.all_results)
        # Collect Elits 
        # print(population, type(population), vals, type(vals))
        if self.operation == 'maximization':
            elists = list(old_pop[-amount_of_elits:])
           
        else:
            elists = list(old_pop[:amount_of_elits])
            
        probs = vals
            # probs = [-value_i for value_i in vals]
        # eps should decrease with time steps 
        
        new_population = list(elists) 
        self.mutations = []

        for i in range(amount_of_mutation): 
            mutant = self.create_mutation(population,eps = eps, std = std)
            self.mutations.append(mutant)
        

        crossovers = self.create_crossover(population, probs, amount_of_crossover)

            
        
        # concatante 
        new_population += list(self.mutations) 
        new_population += list(crossovers)

        return new_population
    
    
    def create_crossover(self, population, vals, amount_of_crossover):
        n = self.population_size
        sum_vals = np.sum(vals)
        probs = [val / sum_vals for val in vals] 
 
        if self.operation == 'maximization':
            pass
        else:
            probs1 = [1 - p for p in probs]
            sum_p = sum(probs1) 
            probs = [p / sum_p for p in probs1]
 
        if self.options['technique'] == 'RankingSelection':
            beta = 0 # can be selected from [0,2] # 0 be proporation to the rank 2 is for the opposite 
            
            if self.operation == 'maximization':
                probs_rank = np.arange(self.population_size) + 1
            else:
                probs_rank = np.arange(self.population_size,0, -1)
                
            probs = 1/self.population_size * (beta - 2* (beta -1) * (probs_rank  -1)/(self.population_size-1)) 
            
            
        
        cross_over_list = []
        
        for i in range(amount_of_crossover):
            # time.sleep(3)
            indexs = np.random.choice(n, size = 2, p = probs, replace = False)

            chrom1 = population[indexs[0]]
            chrom2 = population[indexs[1]]
            chromosomes = [chrom1, chrom2]#chromosome1, chromosome2
            
            sum_p1_p2 = probs[indexs[0]] + probs[indexs[1]]
            p1 = probs[indexs[0]]/sum_p1_p2
            p2 = 1 - p1
            
            cross_i = []
            
            for g in range(self.number_of_genes):
                if self.parameters[g] == 'discrete':
                    index = np.random.choice([0,1], size = 1, p = [p1,p2])[0]

                    cross_i.append(chromosomes[index][g])
                    
                else: # for continues
                    Beta = 0.
                    alpha = (1 + 2*Beta)*np.random.random() - Beta;
                    # alpha = 0.1 + np.random.random()*(sensitivity-0.1)
                    
                    value_g = chromosomes[0][g] + alpha*(chromosomes[1][g] - chromosomes[0][g]) 
                    cross_i.append(value_g)
                    
            cross_over_list.append(cross_i)
            
        return cross_over_list
                    
            
                
        
        
    def update_std(self, std, step):
        # go down to 0.02 - in linear fashion 
        std = (0.02 - self.std_0) / self.max_iteration *step + self.std_0
        std = max(0.02, std)
        return std
        
    def udate_eps(self, eps,step):
        # goes linearly to 0.05
        eps = (0.05 - self.eps_0) / self.max_iteration *step + self.eps_0
        eps = max(0.05, eps)
        return eps
    
  
    def optimize(self, parameters_types, search_range, fitness_fucntion, max_iteration = 2, population_size = 3,
                 elits = 0.2, mutation = 0.2, operation = 'maximization', initial_population = None ,
                 load_checkpoint= False, path_to_checkpoint = None, verbose = True):
        """
        parameters = a list of types, can be discrete or continue (gene tpyes)
        search_range = is a list of a list
                        is the parameter is a continue type, then the list will be consist of 
                        [min value, max balue] and if it is discreate then it is the set of options
    

        """
        t0 = time.time()
        self.fitness_fucntion = fitness_fucntion
        self.parameters = parameters_types
        self.number_of_genes = len(parameters_types)
        self.search_range = search_range
        self.operation = operation
        self.population_size = population_size
        
        if operation == 'maximization':
            self.best_result = -sys.maxsize
        else:
            # for minimization
            self.best_result = sys.maxsize
            
            
        self.elits = elits
        self.mutation = mutation
        
        self.population_size = population_size
        # print(population_size)
        self.max_iteration = max_iteration
        
        if initial_population is None:
            population = self.create_population(population_size = self.population_size)
        else:
            population = initial_population

        self.step = 1
        eps = 1.0
        std = 2.0
        self.all_results = []
        ids = []
        self.std_0 = std
        self.eps_0 = eps
        self.vals = [None for i in range(self.population_size)]
        
        if load_checkpoint:
            population = self.load_checkPoint(path_to_checkpoint = path_to_checkpoint)
            
            for i in range(population_size):
                ids.append(i)
                
            
            [print(k1,k2) for k1,k2 in zip(population, self.vals)]
                
            # population = self.update_population(population,self.vals,ids, 
            #                                     eps = self.udate_eps(eps,self.step), 
            #                                     std = self.update_std(std,self.step))
            
            

        while self.step <= self.max_iteration:
        # Activate the workers
            # if self.step%5 == 0:
            print("Generation:", self.step)
            self.last_population = population.copy()
            # self.last_vals = self.vals.copy()
            if self.step > 2:
                # old_best_res = float(self.best_result)
                self.eps_current = self.udate_eps(eps,self.step)
                self.std_current = self.update_std(std,self.step)
                population = self.update_population(population,self.vals,ids, 
                                                    eps = self.eps_current , 
                                                    std = self.std_current)
            
            t0 = time.time()   
            for i in range(population_size):
                data = population[i]
                self.worker(data,i)
                ids.append(i)  
                print("Sleeping 5 sec")
                time.sleep(5)
            print("Time per generation in sec:", time.time() - t0) 
           
            if (self.step) % 1 == 0 and verbose == True:
                print("Iteration:", self.step, "Best result:", self.best_result, "Best chromosome:", self.best_chromosome,
                      " eps:", self.eps_current, " std:", self.std_current )            
                for i, j in zip(population, self.vals):
                    print("Pop:", i, "value:", j)
            self.step += 1
            
            
            
            if self.step == 2:   
                vals = self.vals
                # print(vals, population,ids)
                vals, population,ids = zip(*sorted(zip(vals,population,ids), key = lambda x: x[0]))
                vals = list(vals)
                population = list(population)
                ids = list(ids)
                #Update best result
                
                if self.operation == 'maximization':
                    if vals[-1] > self.best_result:
                        self.best_result = vals[-1]
                        self.best_chromosome = population[-1]
                else:
                    if vals[0] < self.best_result:
                        self.best_result = vals[0]
                        self.best_chromosome = population[0]
                        
            self.image_and_saving()
        print("Time:", time.time() - t0)
        return self.last_population, self.vals, self.best_result, self.best_chromosome, self.all_results
        
    def image_and_saving(self):
        if not os.path.isdir('GA_checkpoints'):
            os.makedirs('GA_checkpoints')
            
        name = 'GA_checkpoints/checkPoint_' + str(self.step) + '.pk'
        checkPoint= [self.last_population, self.vals, self.best_result, self.best_chromosome, self.all_results]
        
        with open(name, 'wb') as file:
            pickle.dump(checkPoint, file)
        
        ### Create an image 
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis

        # plt.rcParams['font.size'] = 12
        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams["font.serif"] = ["Times New Roman"]
        ax.set_xlabel('#Generation')
        ax.set_ylabel('Ft(x)')
        ax.scatter(np.arange(len(self.all_results)), self.all_results)
        name = 'GA_checkpoints/' + str(self.step) + '.png'
        
        fig.savefig(name)
        plt.close(fig)  
        
    def worker(self, data, i):
        # i is for changing the coresponding location in the array which belong to the specific chromosom
        print("Worker: ", i, " Started!")
        value = self.fitness_fucntion(data, i)
        
        if type(value) == tuple or type(value) == list:
            value = value[0]
            
        mutex.acquire()
        if value == None:
            value = -sys.maxsize
        self.vals[i] = value
        mutex.release()
        print("Worker: ", i, " Finished!", "Vals:", value)
    
    def load_checkPoint(self, path_to_checkpoint = None):
        if path_to_checkpoint is None:
            # Extract the latest file in the path
            folder = 'GA_checkpoints'
            assert os.path.isdir(folder), 'There is no ' + folder + ' folder!!!'
            
            path_to_folder = 'GA_checkpoints/'
            files = os.listdir(path_to_folder)
            
            files = sorted(os.listdir(path_to_folder), key = lambda t: os.stat(path_to_folder + t).st_mtime) 
            files = list(filter(lambda x: 'checkPoint' in x, files))
            checkPoint_file = files[-1]
            
            path_to_checkpoint = path_to_folder + checkPoint_file
            
            
        with open(path_to_checkpoint, 'rb') as file:
            checkpoint = pickle.load(file)
                   
        [self.last_population,
         self.vals,
         self.best_result,
         self.best_chromosome,
         self.all_results] = checkpoint 
        
        self.step = extract_number_form_name(path_to_checkpoint) + 1


        print(path_to_checkpoint, 'was load successfully as the checkpoint file!')
        
        return list(self.last_population)
# def func(x):
#     x = tf.convert_to_tensor(x)
#     y = tf.math.square(x[0]) + (x[1])**2 + (x[2])**2
#     return - y  

# if __name__ =='__main__':

#     t0 = time.time()
#     options = {}
#     options['technique'] = 'RankingSelection'
#     ga = Genetic_algorithm(options=options)
#     pop, val, best_res, best_solution, all_results = ga.optimize(parameters_types = ['continouse','continouse', 'continouse'],
#                                                                  search_range = [[-5,5],[-100,100], [-100,100]],
#                                                                  fitness_fucntion=func , population_size =10, 
#                                                                  operation = 'maximization', max_iteration = 600,
#                                                                  verbose= False)
#     # print("res:", val)#, best_res, best_solution)
#     print("Time:", time.time() - t0)
#     print("Best Solution:", best_solution)
#     plt.figure()
#     plt.plot(all_results)
#     plt.show()
    
#     # print("all results:", all_results)
            
    