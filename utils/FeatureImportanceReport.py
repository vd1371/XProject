import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import OrderedDict
import pprint
import seaborn as sns
import re

def report_feature_importance(direc,
                            features_vals,
                            X,
                            Y,
                            n_top_features,
                            label = "Test",
                            logger = None,
                            should_plot_heatmap = False):

    print ("About to conduct feature importance")
    features_names = list(X.columns)
    
    best_features = dict(zip(features_names, abs(features_vals)))
    features_ = pd.Series(OrderedDict(sorted(best_features.items(), key=lambda t: t[1], reverse =True)))
    
    L = len(features_names)

    logger.info(f"Feature importance based on {label}\n" + pprint.pformat(features_.nlargest(min(L, n_top_features))))

    plt.clf()
    ax = features_.nlargest(n_top_features).sort_values(ascending=True).plot(kind='barh', title = label)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(direc + "/"+ f'{label}-FS.png')
    del fig
    plt.close()

    if should_plot_heatmap:
        df = X.copy()
        df[Y.name] = Y
        df['Counter'] = np.ones(len(df))

        for feat in features_.index[:n_top_features]:

            # sns.set(font_scale=20)
            plt.figure(figsize=(2.5, 2.5))
            piv = df.pivot_table(values = 'Counter', index = feat, columns = Y.name, aggfunc='sum')

            #plot heat map
            sns.set(font="Times New Roman")
            grid = sns.heatmap(piv,
                                annot=False,
                                cmap="RdYlGn",
                                # annot_kws={"size": 500},
                                # cbar_kws={"shrink": 1},
                                square = True)
            
            # grid.tick_params(labelsize = 300)
            # grid.set_xlabel("Features" ,fontsize=300, labelpad= 600)
            # grid.set_ylabel("Features" ,fontsize=300, labelpad= 600)
            
            fig = grid.get_figure()
            plt.tight_layout()

            file_name = re.sub('[^\w\-_\. ]', '_', f'HeatMap-{label}-{feat}.tiff')
            fig.savefig(direc + "/" + file_name, dpi = 300, bbox_inches = 'tight')
            del fig
            plt.clf()
            plt.close("all")
            
            mpl.style.use('classic')


    