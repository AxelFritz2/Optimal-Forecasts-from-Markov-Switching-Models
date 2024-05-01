import sys
sys.path.append("./src")

import pandas as pd
from data_preparation import DataPreparation
from AR_MRS import AR_SC_Markov
from MRS import SC_Markov
from AR_MRS_OW import AR_SC_Markov_OW


if __name__ == "__main__":
    df = pd.read_csv("data/GNP.csv")

    print(125 * '*')
    print(50*'*' + ' Préparation des Données ' + 50*'*')
    print(125 * '*')
    dataprep = DataPreparation(df)

    dataprep.prepare_data()
    dataprep.plot_GNP_crisis()

    print(dataprep.df)

    print("\n")
    print(111*'*')
    print(50 * '*' + ' SC MARKOV ' + 50 * '*')
    print(111 * '*')

    model_SC = SC_Markov(dataprep.df, 2)
    model_SC.run_modelization()

    print("\n")
    print(114 * '*')
    print(50*'*' + ' AR SC MARKOV ' + 50*'*')
    print(114 * '*')

    model_AR_SC = AR_SC_Markov(dataprep.df, 2, 2)
    model_AR_SC.run_modelization()

    print("\n")
    print(135 * '*')
    print(50*'*' + ' AR SC MARKOV with Optimal Weigths ' + 50*'*')
    print(135 * '*')

    model_AR_SC_OW = AR_SC_Markov_OW(dataprep.df, 2, 2)
    model_AR_SC_OW.run_modelization()





