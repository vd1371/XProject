import os, sys
parent_dir = os.path.split(os.path.dirname(__file__))[0]
sys.path.insert(0,parent_dir)
from Reporter import *

from operator import itemgetter

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

        
class Trees(Report):
    
    def __init__(self, df,
                        name = None,
                        split_size = 0.2,
                        should_shuffle = True,
                        k = 5,
                        num_top_features = 10,
                        is_imbalanced = False):
        
        super(Trees, self).__init__(name, 'Trees')
        
        self.num_top_features = num_top_features
        self.k = k
        self.is_imbalanced = is_imbalanced
        
        self.X_train, self.X_test, self.Y_train, self.Y_test, self.dates_train, self.dates_test = prepare_data_simple(df, split_size, False, should_shuffle,
                                                                                                                      is_imbalanced = is_imbalanced)
        
        
        if not is_imbalanced:
            self.X = pd.concat([self.X_train, self.X_test], axis = 0, join='outer')
            self.Y = pd.concat([self.Y_train, self.Y_test], axis = 0, join='outer')
    
    
    def setParams(self, n_estimators = 2000, bootstrap = True,
                 max_depth = 5, max_features = 2, min_samples_leaf=4, min_samples_split = 0.6,
                 should_cross_val=True, n_jobs = 1, verbose = 0):
        
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.should_cross_val = should_cross_val
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def setTrees(self, trees_dict = {'RF':RandomForestClassifier}):
        self.trees = trees_dict
    
    @timeit
    def fit(self, tree, model_name):
        
        self.log.info("\n######################################################################################################################")
        self.log.info(f'---{model_name} with {self.n_estimators} estimators is about to fit')
        
        if model_name != 'DT':
            model = tree(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                        max_features=self.max_features, bootstrap=self.bootstrap,
                        n_jobs=self.n_jobs,  verbose=self.verbose)
        else:
            model = tree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                        max_features=self.max_features)
        
        model.fit(self.X_train, self.Y_train)
        print (f"{model_name} is fitted")
        
        if self.should_cross_val and not self.is_imbalanced:
            scores = cross_val_score(model, self.X, self.Y, cv=self.k, verbose=0)
            self.log.info(f"---- Cross validation with {self.k} groups----\n\nThe results on each split" + str(scores)+"\n")
            self.log.info(f"The average of the cross validation is {np.mean(scores):.2f}\n")
            
            print (f"Cross validation is done for {model_name}. Score: {np.mean(scores):.2f}")
        
                
        self.evaluate_classification(self.Y_train, model.predict(self.X_train), self.dates_train, model_name+'-OnTrain')
        self.evaluate_classification(self.Y_test, model.predict(self.X_test), self.dates_test, model_name+'-OnTest')
        
        # Plotting the Importances
        feature_importances_ = {}
        for i in range(len(model.feature_importances_)):
            feature_importances_[self.X_test.columns[i]] = model.feature_importances_[i]
        
        self.report_feature_importance(feature_importances_, self.num_top_features, label = model_name )
    
    def tune(self, model, model_name,
                        grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                                'max_features': ['auto', 'sqrt'],
                                'max_depth': [int(x) for x in np.linspace(5, 20, num = 15)] + [None],
                                'min_samples_split': [2, 5, 10],
                                'min_samples_leaf': [1, 2, 4],
                                'bootstrap': [True, False]},
                                should_random_search = False,
                                n_iter = 1):

            
        self.log.info(f'------ {model_name} is going to be Tunned with \n -----')
        model = self.trees[model_name]()
        if should_random_search:
            search_models = RandomizedSearchCV(estimator = model, param_distributions = grid, n_iter = n_iter, cv = self.k, verbose=2, n_jobs = self.n_jobs)
        else:
            search_models = GridSearchCV(estimator = model, param_distributions = grid, cv = self.k, verbose=2, n_jobs = self.n_jobs)
        
        search_models.fit(self.X, self.Y)
         
        self.log.info(f"\n\nBest params:\n{pprint.pformat(search_models.best_params_)}\n")
        self.log.info(f"\n\nBest score: {search_models.best_score_:0.4f}\n\n")
        print (search_models.best_score_)
    
    @timeit
    def tuneTrees(self, grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                                'max_features': ['auto', 'sqrt'],
                                'max_depth': [int(x) for x in np.linspace(5, 20, num = 15)] + [None],
                                'min_samples_split': [2, 5, 10],
                                'min_samples_leaf': [1, 2, 4],
                                'bootstrap': [True, False]},
                                should_random_search = False,
                                n_iter = 1):
        
        for model_name in self.trees:
            
            self.log.info(f'------ {model_name} is going to be Tunned with \n -----')
            tree = self.trees[model_name]()
            if should_random_search:
                search_models = RandomizedSearchCV(estimator = tree, param_distributions = grid, n_iter = n_iter, cv = self.k, verbose=2, n_jobs = self.n_jobs)
            else:
                search_models = GridSearchCV(estimator = tree, param_grid = grid, cv = self.k, verbose=2, n_jobs = self.n_jobs)
           
            search_models.fit(self.X, self.Y)
            
            self.log.info(f"\n\nBest params:\n{pprint.pformat(search_models.best_params_)}\n")
            self.log.info(f"\n\nBest score: {search_models.best_score_:0.4f}\n\n")
            print (search_models.best_score_)
            
    def goTrees(self):
        
        for model_name in self.trees:
            print (f"{model_name} is about to start")
            tree = self.trees[model_name]
            self.fit(tree, model_name)
        
@timeit
def run():
    file = 'Digit_train'
    
    tr = Trees(file, name = file, split_size=0.2, should_shuffle=True, k=5, num_top_features = 20, is_imbalanced=False)
    
    tr.setParams(n_estimators = 20000,
                 bootstrap = True,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=2,
                 max_features=None,
                 should_cross_val=False,
                 n_jobs = -1, verbose = 1)
    
    
#         tr.setTrees({'RF':RandomForestClassifier, 'ETC':ExtraTreesClassifier, 'DT':DecisionTreeClassifier})
    tr.setTrees({'ETC':ExtraTreesClassifier})

#     tr.tuneTrees(grid = {'n_estimators': [int(x) for x in np.arange(50, 1000, 50)],
#                          'max_features': ['auto', 'sqrt'],
#                          'max_depth': [int(x) for x in np.arange(3,20,3)] + [None],
#                          'min_samples_split': [val for val in np.arange(0.1, 0.9, 0.2)],
#                          'min_samples_leaf': [val for val in np.arange(0.1, 0.5, 0.2)],
#                          'bootstrap': [True, False]},
#                          should_random_search = False,
#                          n_iter = 1)
    tr.goTrees()
    


if __name__ == "__main__":
    run()

        