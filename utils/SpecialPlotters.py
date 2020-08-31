import os
import numpy as np
import matplotlib.pyplot as plt

def train_cv_analyzer_plotter(train_error, cv_error, direc, model_name, xticks = None, should_save_to_csv = False):

    x = [i+1 for i in range(len(train_error))]
    plt.clf()
    plt.xlabel('Steps')
    plt.ylabel('Metrics')
    plt.title(f'Special Analysis')
    plt.plot(x, train_error, label = 'Train_error')
    plt.plot(x, cv_error, label = 'CV Error')
    if not xticks is None:
        plt.xticks(x, xticks)
    plt.legend()
    plt.grid(True)
    plt.savefig(direc + f'/{model_name}-.png')
    plt.show()
    plt.close()

    if should_save_to_csv:
        rprt_df = pd.DataFrame({'X': xticks, 'CV Errors' : cv_error, 'Train Errors' : train_error})
        rprt_df.to_csv(direc+ f'/{model_name}.csv' )