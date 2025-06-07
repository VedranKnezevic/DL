import pandas as pd
import matplotlib.pyplot as plt
import sys
from itertools import product


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("""You need to give 3 arguments to this script 
              1 -> the first or second experiment
              2 -> the name of the hiperparameter you want to visualize
              3 -> the metric you want to visualise (acc or f1)""")
        exit()


    shootout_results = pd.read_csv(f'exp/shootout_{int(sys.argv[1])}.csv')


    if sys.argv[2] == 'rnn_type':
        print('I already use rnn_type for visualisations')
        exit()

    if sys.argv[2] not in shootout_results.columns:
        print(f'This parameter isn\'t in the data in shootout_{int(sys.argv[1])}')
        exit()

    if sys.argv[3] not in ["acc", "f1"]:
        print('I only have metrics acc and f1')
        exit()
    

    hiperparameter = shootout_results[sys.argv[2]].unique()
    rnn_types = shootout_results["rnn_type"].unique()
    metric = shootout_results.groupby(by=["rnn_type", sys.argv[2]]).mean()[sys.argv[3]]
    

    for rnn_type in rnn_types:
        plt.plot(hiperparameter, metric[rnn_type], label=rnn_type)
        
    plt.ylabel(f'{sys.argv[3]}')
    plt.xlabel(f'{sys.argv[2]}')
    plt.legend(loc="best")
    plt.show()
