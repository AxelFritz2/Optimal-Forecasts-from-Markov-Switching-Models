import sys
sys.path.append("./src")

import pandas as pd
from data_preparation import DataPreparation
from AR_MRS import AR_SC_Markov
from MRS import SC_Markov
from AR_MRS_OW_KS import AR_SC_Markov_OW
from AR_MRS_OW_US import AR_SC_Markov_OW_US


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
    print(151 * '*')
    print(50*'*' + ' AR SC MARKOV with Optimal Weigths and Known States' + 50*'*')
    print(151 * '*')

    model_AR_SC_OW = AR_SC_Markov_OW(dataprep.df, 2, 2)
    model_AR_SC_OW.run_modelization()

    print("\n")
    print(152 * '*')
    print(50*'*' + ' AR SC MARKOV with Optimal Weigths and Uknown States' + 50*'*')
    print(152 * '*')

    model_AR_SC_OW_US = AR_SC_Markov_OW_US(dataprep.df, 2, 2)
    model_AR_SC_OW_US.run_modelization()






