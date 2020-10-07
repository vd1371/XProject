import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import pprint

def report_feature_importance(direc, features_vals, features_names, n_top_features, label = "Test", logger = None):
        
    print ("About to conduct feature importance")
    best_features = dict(zip(features_names, abs(features_vals)))
    features_ = pd.Series(OrderedDict(sorted(best_features.items(), key=lambda t: t[1], reverse =True)))
    
    L = len(features_names)
    logger.info(f"Feature importance based on {label}\n" + pprint.pformat(features_.nlargest(max(L, n_top_features))))

    plt.clf()
    ax = features_.nlargest(n_top_features).plot(kind='bar', title = label)
    fig = ax.get_figure()
    fig.savefig(direc + "/"+ f'{label}-FS.png')
    del fig
    plt.close()
    