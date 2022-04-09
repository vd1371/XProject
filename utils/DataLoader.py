import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option("display.precision", 2)

import io
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import warnings


class DataLoader():

    def __init__(self, **params):
        print ("About to load the data...")

        self.project_name = params.get("project_name")
        self.split_size = params.get("split_size", 0.2)
        self.random_state = params.get("random_state", None)
        self.should_shuffle = params.get("should_shuffle", True)
        self.is_imbalanced = params.get("is_imbalanced", False)
        self.sampling_strategy = params.get('sampling_strategy', {3: 180000, 2: 120000})
        self.modelling_type = params.get("modelling_type", "r")
        self.k = params.get("k", 5)
        self.n_top_features = params.get("n_top_features", 5)
        self.n_samples = params.get("n_samples", None)
        self.should_log_inverse = params.get("should_log_inverse", False)
        self.should_check_hetero = params.get("should_check_hetero", False)
        self.project_name = params.get("project_name")

        self.file_name, df = params.get("df")
        if isinstance(self.file_name, str):
            self.df = pd.read_csv("./_data_storage/"+ self.file_name + ".csv", index_col = 0)
        
        if isinstance(df, pd.DataFrame):
            self.df = df

        if not self.n_samples == None:
            self.df = self.df.iloc[:n_samples, :]

        if self.should_log_inverse:
            self.df.iloc[:,-1] = np.log(self.df.iloc[:,-1])

        if self.modelling_type == "c":
            self.df.iloc[:, -1], self.encoder = encoded_classes(self.df.iloc[:, -1].values)
            self.classes_ = self.encoder.classes_

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

                for col in get_categorical_cols(self.df):
                    logger.info(str(self.df[col].value_counts()))

                logger.info("Encoder and classes:\n" + \
                                str(dict(zip(range(len(self.encoder.classes_)), self.encoder.classes_))))

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

        for col in self.df.columns[:]:
                self.df[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())

        X, Y, dates = self.df.iloc[:,:-1], self.df.iloc[:,-1], self.df.index

        # X = X[['CPI', 'HSI-Close', 'InterestRate', 'HATPI', 'HATPI-Trend', 'A', 'B', 'ADW-Trend', 'CFA', 'BWTPI']]

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

        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(Y)

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

def encoded_classes(Y):
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    return Y, encoder


def is_in_log_file(address, phrase):

    with open(address) as f:
        f = f.readlines()

    for line in f:
        if phrase in line:
            return True
    return False

def get_categorical_cols(df):
    # This method is called when the categorical columns are not passed by the user

    warning_message = "\n\n---------------------------------------\n" +\
                    "The categorical_columns is not defined by you.\n" +\
                    "I will try to find the categorical columns using heuristic methods.\n" +\
                    "There is a small chance it fails. Consider passing the categorical_columns. \n" +\
                    "---------------------------------------\n"
    warnings.warn(warning_message, UserWarning , stacklevel = 2)

    nominal_dtypes = ['object', 'bool', 'datetime64']

    categorical_columns = []
    for col in df.columns:

        # Adding the string, boolean, and datetimes columns
        if df[col].dtypes in nominal_dtypes:
            categorical_columns.append(col)

        # Checking the integer columns
        elif pd.api.types.is_integer_dtype(df[col].dtypes):
            # A heuristic method to check if the column in categorical
            if df[col].nunique() / df[col].count() < 0.05:
                categorical_columns.append(col)

    return categorical_columns