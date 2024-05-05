import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from tabulate import tabulate


class AR_SC_Markov_OW():
    def __init__(self, df, n_states, order):
        self.df = df
        self.n_states = n_states
        self.order = order

    def initialize_model(self):
        print("Model Initialization : \n"
              "Model : Markov Auto Regression with Optimal Weights and Known States \n"
              f"Number of states : {self.n_states} \n"
              f"order : {self.order}")

        self.model = MarkovAutoregression(self.df['GR_GNP'],
                                          k_regimes = self.n_states,
                                          order = self.order,
                                          switching_ar=False,
                                          switching_variance=True)

    def fit_model(self):
        print("Model Fitting ...")
        self.model_fitted = self.model.fit()
        print("Model Fitted ✅")


    def optimal_weigths(self):
        T = self.df.shape[0]
        pi1 = self.df["crise"].sum() / T
        pi0 = 1 - pi1
        sigma0 = self.df[self.df["crise"] == 0]["GR_GNP"].std()
        sigma1 = self.df[self.df["crise"] == 1]["GR_GNP"].std()
        q = sigma0 / sigma1
        beta0 = self.df[self.df["crise"] == 0]["GR_GNP"].mean()
        beta1 = self.df[self.df["crise"] == 1]["GR_GNP"].mean()
        lamda = ((beta1 - beta0) ** 2) / sigma1 ** 2

        def W00():
            w = 1 / T
            num = 1 + T * (lamda ** 2) * pi1
            den = pi1 * (q ** 2) + pi0 * (1 + T * (lamda ** 2) * pi1)
            w = w * (num / den)
            return w

        def W01():
            w = 1 / T
            num = q ** 2
            den = pi1 * (q ** 2) + pi0 * (1 + T * (lamda ** 2) * pi1)
            w = w * (num / den)
            return w

        def W10():
            w = 1 / T
            num = 1
            den = pi1 * (q ** 2) + pi0 * (1 + T * (lamda ** 2) * pi1)
            w = w * (num / den)
            return w

        def W11():
            w = 1 / T
            num = q ** 2 + T * (lamda ** 2) * pi0
            den = pi1 * (q ** 2) + pi0 * (1 + T * (lamda ** 2) * pi1)
            w = w * (num / den)
            return w

        self.df['Modified_GR_GNP'] = self.df['GR_GNP']

        for i in range(len(self.df) - 1):
            current_crise = self.df.loc[i, 'crise']
            next_crise = self.df.loc[i + 1, 'crise']
            if current_crise == 0 and next_crise == 0:
                self.df.loc[i, 'Modified_GR_GNP'] *= W00()
            elif current_crise == 0 and next_crise == 1:
                self.df.loc[i, 'Modified_GR_GNP'] *= W10()
            elif current_crise == 1 and next_crise == 1:
                self.df.loc[i, 'Modified_GR_GNP'] *= W11()
            elif current_crise == 1 and next_crise == 0:
                self.df.loc[i, 'Modified_GR_GNP'] *= W01()

        self.model = MarkovAutoregression(self.df['Modified_GR_GNP'], k_regimes=2, order=2, switching_ar=False,
                                         switching_variance=True)
        self.model_fitted_OW = self.model.fit()
    def plot_recession_probabilty(self):
        smoothed_marginal_probabilities = self.model_fitted_OW.smoothed_marginal_probabilities
        df_temp = self.df.iloc[self.order:]

        # Plot the series and the smoothed probabilities
        fig, ax1 = plt.subplots()

        ax1.plot(df_temp["DATE"], df_temp['GR_GNP'], label='GR_GNP')
        ax1.set_ylabel('GR_GNP', color='C0')
        ax1.tick_params(axis='y', labelcolor='C0')

        ax2 = ax1.twinx()
        ax2.fill_between(df_temp["DATE"], smoothed_marginal_probabilities[1], step='pre', alpha=0.4, color='C1')
        ax2.set_ylabel('Probability of Regime 1', color='C1')
        ax2.tick_params(axis='y', labelcolor='C1')
        ax2.set_ylim([0, 1])

        fig.tight_layout()
        plt.show()

    def get_transition_matrix(self):
        p00, p10, const0, const1, sigma0, sigma1, ar1_L1, ar1_L2 = self.model_fitted_OW.params

        log_odds = np.array([const0, const1])
        diff_log_odds = np.array(
            [log_odds[i] - log_odds[j] for i in range(len(log_odds)) for j in range(len(log_odds))]
        )

        transition_probs = np.exp(diff_log_odds) / (1 + np.exp(diff_log_odds))

        transition_matrix = np.array(
            [transition_probs[i:i + len(log_odds)] for i in range(0, len(transition_probs), len(log_odds))]
        )

        transition_matrix /= np.sum(transition_matrix, axis=1)[:, None]
        transition_matrix = self.model_fitted.regime_transition
        print("\n")
        print("Transition Matrix : ")
        print("      | Régime 0 | Régime 1")
        print("--------------------------------")
        for i, row in enumerate(transition_matrix):
            print(f"De R{i} | {row[0]} | {row[1]}")
        print("\n")

    def get_metrics(self):
        predictions = self.model_fitted_OW.predict()
        actual = self.df['Modified_GR_GNP'].iloc[4:]

        liste_periode = ['1947-01-01 à 2023-10-01 ', '1947-01-01 à 1996-10-01 ', '1997-01-01 à 2008-10-01',
                         '2009-01-01 à 2017-10-01', '2018-01-01 à 2023-10-01']

        # Liste pour stocker les résultats
        results = []

        # Boucler sur les périodes définies
        for periode in liste_periode:
            start_date, end_date = periode.split(" à ")
            part_actual = actual[(self.df['DATE'] >= start_date) & (self.df['DATE'] <= end_date)]
            part_predictions = predictions[(self.df['DATE'] >= start_date) & (self.df['DATE'] <= end_date)]

            # Calculer les métriques pour cette partie
            mse = mean_squared_error(part_actual, part_predictions)
            mae = mean_absolute_error(part_actual, part_predictions)

            # Ajouter les résultats à la liste
            results.append([periode, mse, mae])

        # Afficher les résultats dans le terminal
        headers = ["Period", "MSE", "MAE"]
        print("Tableau d'Erreurs")
        print(tabulate(results, headers=headers, floatfmt=".4f"))
        print("\n")

    def run_modelization(self):
        self.initialize_model()
        self.fit_model()
        self.optimal_weigths()
        self.plot_recession_probabilty()
        self.get_transition_matrix()
        self.get_metrics()
