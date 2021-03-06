import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader():

    def __init__(self, df,
                        split_size = 0.2,
                        should_shuffle=True,
                        is_imbalanced=False,
                        random_state = None,
                        k = 5,
                        n_top_features = 5,
                        n_samples = None,
                        should_log_inverse = False,
                        modelling_type = 'regression'):

        self.split_size = split_size
        self.random_state = random_state
        self.should_shuffle = should_shuffle
        self.is_imbalanced = is_imbalanced
        self.modelling_type = modelling_type

        self.should_log_inverse = should_log_inverse

        self.k = k
        self.n_top_features = n_top_features

        if isinstance(df, str):
            self.df = pd.read_csv("./_data_storage/"+ df + ".csv", index_col = 0)
        
        elif isinstance(df, pd.DataFrame):
            self.df = df

        else:
            raise ValueError ("Unsupported input format for dataloader. It should be str or DataFrame")

        if not n_samples is None:
            self.df = self.df.iloc[:n_samples, :]

        if should_log_inverse:
            self.df.iloc[:,-1] = np.log(self.df.iloc[:,-1])

        print ("data is loaded")


    def load_with_test(self):

        X, Y, dates = self.df.iloc[:,:-1], self.df.iloc[:,-1], self.df.index
        X_train, X_test, Y_train, Y_test, \
            dates_train, dates_test = train_test_split(X, Y, dates,
                                                        test_size = self.split_size,
                                                        shuffle = self.should_shuffle,
                                                        random_state = self.random_state)
        if self.is_imbalanced:
            X_train, Y_train, dates_train = self._over_sample(X_train, Y_train, dates_train)

        return X_train, X_test, Y_train, Y_test, dates_train, dates_test

    def load_all(self):
        X, Y, dates = self.df.iloc[:,:-1], self.df.iloc[:,-1], self.df.index
        return X, Y, dates

    def load_with_csv(self):

        X, Y, dates = self.df.iloc[:,:-1], self.df.iloc[:,-1], self.df.index
            
        X_train, X_temp, Y_train, Y_temp, \
            dates_train, dates_temp = train_test_split(X, Y, dates,
                                                        test_size = self.split_size,
                                                        shuffle = self.should_shuffle, 
                                                        random_state = self.random_state)

        X_cv, X_test, Y_cv, Y_test, \
            dates_cv, dates_test = train_test_split(X_temp, Y_temp, dates_temp,
                                                    test_size = 0.5,
                                                    shuffle = self.should_shuffle,
                                                    random_state = self.random_state)
        
        if self.is_imbalanced:
            X_train, Y_train, dates_train = self._over_sample(X_train, Y_train, dates_train)

        return X_train, X_cv, X_test, Y_train, Y_cv, Y_test, dates_train, dates_cv, dates_test

    def _over_sample(self, x_train, y_train, dates):

        print ("Starting to do over sampling")

        if self.modelling_type.lower() == 'c':
            from imblearn.over_sampling import SMOTE

            X_train, Y_train = SMOTE().fit_resample(X_train, Y_train)
            dates_train = ['SMOTE'+str(i) for i in range(len(X_train))]
        
        elif self.modelling_type.lower() == 'r':
            import smogn

            df = x_train.copy()
            df['Y'] = y_train
            df.reset_index(drop = True, inplace = True)
            df_smogn = smogn.smoter(data = df, y = 'Y')

            X_train, Y_train, dates_train = df_smogn.iloc[:, :-1], df_smogn.iloc[:, -1], df_smogn.index

        else:
            raise ValueError ("The modelling_type should be 'r' or 'c' ")

        return X_train, Y_train, dates_train
                
                