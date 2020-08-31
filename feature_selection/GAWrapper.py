import os, sys
sys.path.insert(0,'..')

import time
import numpy as np
import pandas as pd
from datetime import datetime

from Reporter import *
from _Neural import *

from sklearn.utils import shuffle
from deap import tools

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

import pickle

def star_function(x):
    return max(70-x**0.5, 0)**4

def binaryToDecimal(binary_ls): 
    
    decimal = 0
    for i, b in enumerate(reversed(binary_ls)):
        decimal = decimal + b * pow(2, i) 
        i += 1
    
    return decimal

def inTabooList(chrome, taboo_list):
    return binaryToDecimal(chrome) in taboo_list

def initializeGeneration(generation_size, gene_length):
    
    taboo_list = []
    generation = []
    for _ in range(generation_size):
        
        chromosome = list(np.random.randint(2, size=gene_length))
        
        # append the chromosome to 
        taboo_list.append(binaryToDecimal(chromosome))
        
        # Creating a new individual with the created chromosome
        ind = Individual()
        ind.updateChromosome(chromosome)
        generation.append(ind)
    
    return generation, taboo_list

def crossOver(chrome1, chrome2, prob):
    if np.random.random() < prob:
        return tools.cxTwoPoint(chrome1, chrome2)
    return chrome1, chrome2

def mutate(chrome1, chrome2, mutation_percentage = 0.05):
    chrome1, chrome2 = tools.mutFlipBit(chrome1, mutation_percentage), tools.mutFlipBit(chrome2, mutation_percentage)
    return chrome1[0], chrome2[0]

class MLSettings(object):
    def __init__(self):
        super(MLSettings, self).__init__()
    
    def setMLCharacteristics(self, ml_type = 'Regressor',
                             epochs = 2000,
                             activation_func = 'tanh',
                             final_activation_func = 'linear',
                             loss_func = 'mean_squared_error',
                             classes = [1,0],
                             should_shuffle = True,
                             should_early_stop = True,
                             min_delta = 0.0005,
                             patience = 50,
                             batch_size = 50,
                             split_size = 0.4,
                             drop = 1,
                             reg_param = 0.000):
        
        self.ml_type = ml_type
        self.min_delta = min_delta
        self.patience = patience
        self.batch_size = batch_size

        self.activation_func = activation_func
        self.final_activation_func = final_activation_func
        self.loss_func = loss_func
        self.epochs = epochs
        self.should_early_stop = should_early_stop
        self.should_shuffle = should_shuffle
        self.split_size = split_size
        
        self.drop = drop
        self.reg_param = reg_param
        
        self.classes = classes
    
    def setLayers(self, layers):
        self.layers = layers

class Individual(object):
    
    def __init__(self):
        
        super(Individual, self).__init__()
        self.value = 0
    
    def updateChromosome(self, chromosome):
        self.chromosome = chromosome
        
    def __repr__(self):
        return f"Ind-{self.value:.2f}---{self.chromosome}-{binaryToDecimal(self.chromosome)}"
    
    def __str__(self):
        return f"Ind-{self.value:.2f}---{self.chromosome}-{binaryToDecimal(self.chromosome)}"
    
    def updateData(self, data):
        self.data =data
    
    def setValue(self, val):
        self.value = val
    
    def evaluate(self, X, Y, MLS, chromosome):
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle = False)
        self.all_errs = {}
        model = NeuralNet()
        model.setParams(x_train, x_test, y_train, y_test, MLS.ml_type, MLS.layers, MLS.epochs,
                             MLS.activation_func, MLS.final_activation_func, MLS.loss_func,
                             MLS.classes,
                             MLS.should_early_stop,
                             MLS.min_delta, MLS.patience, MLS.batch_size, MLS.split_size,
                             MLS.drop, MLS.reg_param)
            
        self.dnn_train_err, self.dnn_test_err = model.fit_regressor()
        self.all_errs['dnn_train'] = self.dnn_train_err
        self.all_errs['dnn_test'] = self.dnn_test_err
        self.value = star_function(self.dnn_test_err)
        
        # training SVR
        for c in [100000]:
            model = SVR(C=c, kernel= "rbf", epsilon = 0.2, gamma = 'auto')
            model.fit(x_train.values, y_train.values)
            self.all_errs[f'SVR-train-{c}'] = mean_squared_error(y_train, model.predict(x_train.values))
            self.all_errs[f'SVR-test-{c}'] = mean_squared_error(y_test, model.predict(x_test.values))
            
        # training linear regression model
        model = LinearRegression(fit_intercept = True, normalize=False)
        model.fit(x_train.values, y_train.values)
        self.all_errs['Linear-train'] = mean_squared_error(y_train, model.predict(x_train.values))
        self.all_errs['Linear-test'] = mean_squared_error(y_test, model.predict(x_test.values))
#         self.value = star_function(mean_squared_error(y_test, model.predict(x_test.values)))

#         self.all_errs['Test'] = binaryToDecimal(chromosome)
#         self.value = binaryToDecimal(chromosome)
#         self.value = np.random.random()


    def get_all_errs(self):
        return self.all_errs
    
    def get_value(self):
        return self.value

class GA(Report):
    """
    This code will choose the best choices from lots of columns for making a proper model for ML
    It will optimize the cost function of the minimized ML structure
    It will not optimize the structure of the ML, since it will use dropout, it will not be overfitted
    """
    
    def __init__(self, df = pd.DataFrame(),
                 name = None,
                 ml_type = 'Classifier'):
        super(GA, self).__init__(name, 'Wrapper')
        self.ml_type = ml_type
        
        #splitting data into X and Y
        self.X, self.Y, _ = prepare_data_simple(df, split_size=0.4, is_cv = False, is_test = False, should_shuffle = True)
        self.data = pd.concat([self.X, self.Y], axis=1, join = 'inner')
        
        self.all_candidates = list(self.X.columns[:])
        self.gene_length = len(self.all_candidates)
        
#         self.log.info("Genetic optimization for finding the best combination of inputs for neural network Project started\n")
#         self.log.info('\n---Data for %s is loaded---'%self.name)
        
    def shuffleDataset(self):
        self.data = shuffle(self.data) if self.MLS.should_shuffle else self.data
        
        #splitting data into X and Y
        self.X = self.data.iloc[:,:-1].copy()
        self.Y = self.data.iloc[:,-1].copy()
        
    def setMLCharacteristics(self, ml_type = "Regressor",
                             epochs = 2000,
                             activation_func = 'tanh',
                             final_activation_func = 'linear',
                             loss_func = 'mean_squared_error',
                             classes = [0, 1],
                             should_shuffle = True,
                             should_early_stop = True,
                             min_delta = 0.0005,
                             patience = 50,
                             batch_size = 50,
                             split_size = 0.4,
                             drop = 1,
                             reg_param = 0.00):
        
        self.MLS = MLSettings()
        self.MLS.setMLCharacteristics(ml_type, epochs,
                                 activation_func, final_activation_func, loss_func,
                                 classes,
                                 should_shuffle, should_early_stop,
                                 min_delta, patience, batch_size, split_size,
                                 drop, reg_param)
        
        self.shuffleDataset()
        
#         self.log.info('\n---Neural Network properties---\nActivation Function: %s\nFinal Activation Function: %s\nLoss Function: %s\nShould Shuffle: %s\nmin_delta: %0.4f\npatience: %d\nsplit_size: %0.2f\n'%(activation_func,
#                         final_activation_func, loss_func, str(should_shuffle), min_delta, patience, split_size))
        
    
    def setGACharacterisitcs(self, generation_size = 8,
                             mutation_percentage = 0.1,
                             max_generation_number = 200,
                             cores = 4,
                             threshold = 0.01,
                             ga_patience = 10, 
                             elites = 4,
                             crossover_prob = 0.75):
        
        self.cores = cores
        self.generation_size = generation_size
        self.mutation_percentage = mutation_percentage
        self.max_generation_number = max_generation_number
        self.threshold = threshold
        self.ga_patience = ga_patience
        self.elites = elites
        self.crossover_probability = crossover_prob
        
#         self.log.info('\n---Genetic Algorithm properties---\nGenetation Size: %d\nMutation Percentage: %0.1f\nMaximum Generation Number: %d\nCores%d\n'%(generation_size,
#                         mutation_percentage, max_generation_number, cores))
    
    def creatNewGeneration(self, previous_generation, reference_values = [], mutation_percentage = 0.2, taboo_list = []):
        
        new_generation = []
        start = time.time()
        
        for i in range(self.elites):
    
            new_ind = Individual()
            new_ind.updateChromosome(previous_generation[i].chromosome[:])
            new_ind.setValue(previous_generation[i].value)
            new_generation.append(new_ind)
        
        # Creating weight list
        weight_list = []
        for ind in previous_generation:
            weight_list.append(ind.value)
        
        weight_list = np.array(weight_list)/1000
        sum_of_weights = np.sum(weight_list)
        weights = [(i)/sum_of_weights for i in weight_list]
        
        while len(new_generation) < self.generation_size:
            
            # Weighted randomly chosen from the previous generation
            parent1, parent2 = np.random.choice(previous_generation, size = 2, replace = False, p=weights)
            
            # Crossover and mutation
            child1_chromosome, child2_chromosome = crossOver(parent1.chromosome, parent2.chromosome, self.crossover_probability)
            child1_chromosome, child2_chromosome = mutate(child1_chromosome, child2_chromosome, mutation_percentage = self.mutation_percentage)
            
            if not inTabooList(child1_chromosome, taboo_list) and 1 in child1_chromosome:
                new_ind = Individual()
                new_ind.updateChromosome(child1_chromosome)
                taboo_list.append(binaryToDecimal(child1_chromosome[:]))
                new_generation.append(new_ind)
                
            if not inTabooList(child2_chromosome, taboo_list) and 1 in child2_chromosome:
                new_ind = Individual()
                new_ind.updateChromosome(child2_chromosome)
                taboo_list.append(binaryToDecimal(child2_chromosome[:]))
                new_generation.append(new_ind)
                
        
        # if possibly one more child is added to the population
        if len(new_generation) > self.generation_size:
            new_generation = new_generation[:self.generation_size]
        
#         print('Taboo list length', len(taboo_list))
#         print ('New generation is created in: %0.2f seconds' %(time.time()-start))
        
        return new_generation

    def analyzeGeneration(self, generation):
        
        gener = generation[:]
        
        start = time.time()
        def getCols(cols, chromosome):
            ls = []
            for i in range(len(chromosome)):
                if chromosome[i] == 1:
                    ls.append(cols[i])
            return ls
        
        rprt = ""
        for ind in generation:
            '''
            ml_type, X, Y, layers, epochs,
                        activation_func, final_activation_func, loss_func,
                        should_early_stop,
                        min_delta, patience, batch_size, split_size)
            '''
            
            cols = getCols(self.all_candidates, ind.chromosome)
            
            X = self.X[cols]
            Y = self.Y
            
            layers = [2*len(cols) for i in range(3)]
            self.MLS.setLayers(layers)
            
            ind.evaluate(X, Y, self.MLS, ind.chromosome)
#             print (f"Ind {generation.index(ind)+1}/{self.generation_size} evaluated.")
#             rprt += f"\n{generation.index(ind)+1}: Ind with chrome:{cols} value = {ind.value}"
            
#         self.log.info(rprt)
        gener.sort(key=lambda x:x.value, reverse = True)
        
#         print ('New generation is analyzed in: %0.2f seconds' %(time.time()-start))
#         self.log.info('New generation is analyzed in: %0.2f seconds' %(time.time()-start))
        
        return gener
    
    def optimize(self):
        
        # Loading the 
        if os.path.exists(f'{self.directory}/generation'):
            with open(f'{self.directory}/generation', 'rb') as generation_file:
                new_generation = pickle.load(generation_file)
            with open(f'{self.directory}/taboo_list', 'rb') as taboo_file:
                taboo_list = pickle.load(taboo_file)
        else: 
            new_generation, taboo_list = initializeGeneration(self.generation_size, self.gene_length)
        
        #2 Define the primitive values
        minimum_value = -999999
        generation_number = 0
        
        while generation_number < self.max_generation_number:
            
            # Condition of ending the optimization if there are no signifanct improve
#             if len(best_value_list) > self.ga_patience and abs(best_value_list[-1] - best_value_list[-self.ga_patience])<self.threshold:
#                 break

            # Increasing the generation number
            generation_number += 1

            # Analyze Generation
            analyzed_generation = self.analyzeGeneration(new_generation)
            
            start = time.time()
            # loading the file for errors
            dir = self.directory + '/GAWrapper-All_errors-' + self.name + '.csv'
            if os.path.exists(dir):
                df = pd.read_csv(dir, index_col = 0)
            else:
                df = pd.DataFrame(columns = list(analyzed_generation[0].get_all_errs().keys()))
            generation_number = len(df)
            print (f'Generation:{generation_number}')
            
            # Finding the best value and adding to the best value
            best_value = analyzed_generation[0].get_value()
            print (analyzed_generation[0])
            self.log.info(f"So far The best option is: {analyzed_generation[0]} - Generation {generation_number}")
            
            # saving the error file
            df.loc[generation_number] = list(analyzed_generation[0].get_all_errs().values())
            df.to_csv(dir)
        
            # Create new generation
            new_generation = []
            new_generation = self.creatNewGeneration(analyzed_generation,
                                                    self.all_candidates,
                                                    self.mutation_percentage,
                                                    taboo_list)
            
            
            with open(f'{self.directory}/generation', 'wb') as generation_file:
                pickle.dump(new_generation, generation_file)
            with open(f'{self.directory}/taboo_list', 'wb') as taboo_file:
                pickle.dump(taboo_list, taboo_file)
            
            # shuffling the dataset if needed
            if self.MLS.should_shuffle: self.shuffleDataset()
        
def geneticOptimization():
    
    myGA = GA('Diabetes1', name = 'Diabetes1')
    myGA.setGACharacterisitcs(generation_size = 100, mutation_percentage = 0.05, max_generation_number = 1,
                              threshold=0.01, ga_patience = 20, elites = 5)
    
    myGA.setMLCharacteristics(ml_type = 'Regressor', epochs = 2000, activation_func = 'tanh', final_activation_func='linear',
                              loss_func= 'mean_squared_error', should_shuffle=False, should_early_stop = True,
                              min_delta = 2, patience = 100, batch_size=64, split_size=0.2,
                              drop = 0.5, reg_param = 0.0000)
    
    myGA.optimize()
    
        
if __name__ == "__main__":
#     cProfile.run('geneticOptimization()')
    
    geneticOptimization()    
    