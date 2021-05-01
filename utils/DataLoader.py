import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option("display.precision", 2)
import io
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


class DataLoader():

    def __init__(self, **params):
        print ("About to load the data...")

        df = params.pop("df")
        self.split_size = params.pop("split_size", 0.2)
        self.random_state = params.pop("random_state", None)
        self.should_shuffle = params.pop("should_shuffle", True)
        self.is_imbalanced = params.pop("is_imbalanced", False)
        self.sampling_strategy = params.pop('sampling_strategy', {3: 180000, 2: 120000})
        self.modelling_type = params.pop("modelling_type", "r")
        self.k = params.pop("k", 5)
        self.n_top_features = params.pop("n_top_features", 5)
        self.n_samples = params.pop("n_samples", None)
        self.should_log_inverse = params.pop("should_log_inverse", False)

        if isinstance(df, str):
            self.df = pd.read_csv("./_data_storage/"+ df + ".csv", index_col = 0)
        
        elif isinstance(df, pd.DataFrame):
            self.df = df

        else:
            raise ValueError ("Unsupported input format for dataloader. It should be str or DataFrame")

        if not self.n_samples is None:
            self.df = self.df.iloc[:n_samples, :]

        if self.should_log_inverse:
            self.df.iloc[:,-1] = np.log(self.df.iloc[:,-1])

        if self.should_shuffle:
            self.df = self.df.sample(frac = 1, random_state = self.random_state)

        print ("Data is loaded...")

    def log_description(self, logger):

        self.log = logger

        if not is_in_log_file(logger.address, '*Data Description*'):
            self.log.info('*Data Description*\n')

            buf = io.StringIO()
            self.df.info(buf=buf)
            logger.info(buf.getvalue())
            logger.info(str(self.df.describe()))

            if self.modelling_type.lower() == 'c':
                for col in self.df.columns:
                    logger.info(str(self.df[col].value_counts()))

            print ("Data description is logged...")
        
        else:
            print ("Data description is already logged")

        logger.info(f"Split size: {self.split_size}\n"
                    f"random_state: {self.random_state}\n"
                    f"should_shuffle: {self.should_shuffle}\n"
                    f"is_imbalanced: {self.is_imbalanced}\n"
                    f"sampling_strategy: {self.sampling_strategy}\n"
                    f"modelling_type: {self.modelling_type}\n"
                    f"should_log_inverse: {self.should_log_inverse}\n"
                    f"k for cross_validation: {self.k}\n"
                    f"n_top_features: {self.n_top_features}\n"
                    f"n_samples: {self.n_samples}\n")

    def load_all(self):
        X, Y, dates = self.df.iloc[:,:-1], self.df.iloc[:,-1], self.df.index
        return X, Y, dates

    def load_with_test(self):

        X, Y, dates = self.load_all()
        X_train, X_test, Y_train, Y_test, \
            dates_train, dates_test = train_test_split(X, Y, dates,
                                                        test_size = self.split_size,
                                                        shuffle = self.should_shuffle,
                                                        random_state = self.random_state)
        if self.is_imbalanced:
            X_train, Y_train, dates_train = self._over_sample(X_train, Y_train, dates_train)

        return X_train, X_test, Y_train, Y_test, dates_train, dates_test

    def load_with_csv(self):

        X, Y, dates = self.load_all()

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

    def load_for_classification(self):

        X, Y, dates = self.load_all()

        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(encoded_Y)

        # Splitting the original Y
        Y_original_train, Y_original_temp = train_test_split(Y,
                                                            test_size = self.split_size,
                                                            shuffle = self.should_shuffle,
                                                            random_state = self.random_state)
        Y_original_cv, Y_original_test = train_test_split(Y_original_temp,
                                                        test_size = 0.5,
                                                        shuffle = self.random_state)
         
        #splitting data into train, cross_validation, test
        X_train, X_temp, Y_train, Y_temp, \
            dates_train, dates_temp = train_test_split(X, dummy_y, dates,
                                                        test_size = self.split_size,
                                                        shuffle = self.should_shuffle,
                                                        random_state = self.random_state)
        X_cv, X_test, Y_cv, Y_test, \
            dates_cv, dates_test = train_test_split(X_temp, Y_temp, dates_temp,
                                                    test_size = 0.5,
                                                    shuffle = self.should_shuffle,
                                                    random_state = self.random_state)
        if self.is_imbalanced:
            X_train, Y_original_train, dates_train = self._over_sample(X_train, Y_original_train, dates_train)
            Y_train = encoder.transform(Y_original_train)
            Y_train = np_utils.to_categorical(Y_train)

        return X_train, X_cv, X_test, \
                    Y_train, Y_cv, Y_test, \
                        Y_original_train, Y_original_cv, Y_original_test, \
                            dates_train, dates_cv, dates_test

    def _over_sample(self, X_train, Y_train, dates):

        print ("About to handle imbalanced dataset...")

        if self.modelling_type.lower() == 'c':

            from imblearn.over_sampling import SMOTE
            from imblearn.over_sampling import RandomOverSampler
            from imblearn.under_sampling import RandomUnderSampler

            # X_train, Y_train = RandomUnderSampler(sampling_strategy = self.sampling_strategy,
                                                    # random_state = self.random_state).fit_resample(X_train, Y_train)
            X_train, Y_train = SMOTE(random_state = self.random_state).fit_resample(X_train, Y_train)
            # X_train, Y_train = RandomOverSampler(random_state = self.random_state).fit_resample(X_train, Y_train)

            dates_train = ['Imb'+str(i) for i in range(len(X_train))]
        
        elif self.modelling_type.lower() == 'r':
            import PyImbalReg as pir

            df = x_train.copy()
            df['Y'] = y_train
            df = pir.GNHF(df = df,
                            perm_amp = 0.01,
                            bins = 50,
                            should_log_transform = True,
                            random_state = self.random_state).get()

            X_train, Y_train, dates_train = df.iloc[:, :-1], df.iloc[:, -1], df.index

        else:
            raise ValueError ("The modelling_type should be 'r' or 'c' ")

        print ("Imbalanced data is handled now...")
        return X_train, Y_train, dates_train
    
    def summary(self, logger):
        '''Log the descrive and info of the dataset'''
        logger.info(self.df.info())
        logger.info(self.df.describe())


def is_in_log_file(address, phrase):

    with open(address) as f:
        f = f.readlines()

    for line in f:
        if phrase in line:
            return True
    return False