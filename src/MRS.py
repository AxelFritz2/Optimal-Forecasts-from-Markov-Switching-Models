import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from tabulate import tabulate


class SC_Markov():
    def __init__(self, df, n_states):
        self.df = df
        self.n_states = n_states

    def initialize_model(self):
        print("Model Initialization : \n"
              "Model : Markov Regression \n"
              f"Number of states : {self.n_states} \n")

        self.model = MarkovRegression(self.df['GR_GNP'],
                                      k_regimes = self.n_states,
                                      trend='c',
                                      switching_variance=True)

    def fit_model(self):
        print("Model Fitting ...")
        self.model_fitted = self.model.fit()
        print("Model Fitted ✅")

    def get_summary(self):
        print(self.model_fitted.summary())

    def plot_in_sample_forecast(self):
        print("In Sample Forecast in progress ...")
        in_sample_forecasts = self.model_fitted.predict()

        plt.plot(in_sample_forecasts, label='In-sample forecasts')
        plt.plot(self.df['GR_GNP'], label='GNP data')

        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title('In-sample forecasts and train data')

        plt.legend()
        plt.show()

    def plot_recession_probabilty(self):
        smoothed_marginal_probabilities = self.model_fitted.smoothed_marginal_probabilities

        # Plot the series and the smoothed probabilities
        fig, ax1 = plt.subplots()

        ax1.plot(self.df["DATE"], self.df['GR_GNP'], label='GR_GNP')
        ax1.set_ylabel('GR_GNP', color='C0')
        ax1.tick_params(axis='y', labelcolor='C0')
        ax1.set_title('Probabilité de Récession - Modèle MSM')

        ax2 = ax1.twinx()
        ax2.fill_between(self.df["DATE"], smoothed_marginal_probabilities[1], step='pre', alpha=0.4, color='C1')
        ax2.set_ylabel('Probability of Regime 1', color='C1')
        ax2.tick_params(axis='y', labelcolor='C1')
        ax2.set_ylim([0, 1])

        fig.tight_layout()
        plt.show()

    def get_transition_matrix(self):
        p00, p10, const0, const1, sigma0, sigma1 = self.model_fitted.params

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
        predictions = self.model_fitted.predict()

        liste_periode = ['1947-01-01 à 2023-10-01 ', '1947-01-01 à 1996-10-01 ', '1997-01-01 à 2008-10-01',
                         '2009-01-01 à 2017-10-01', '2018-01-01 à 2023-10-01']

        # Liste pour stocker les résultats
        results = []

        # Boucler sur les périodes définies
        for periode in liste_periode:
            start_date, end_date = periode.split(" à ")
            part_actual = self.df[(self.df['DATE'] >= start_date) & (self.df['DATE'] <= end_date)]['GR_GNP']
            part_predictions = predictions[(self.df['DATE'] >= start_date) & (self.df['DATE'] <= end_date)]

            # Calculer les métriques pour cette partie
            mse = mean_squared_error(part_actual, part_predictions)
            mae = mean_absolute_error(part_actual, part_predictions)

            # Ajouter les résultats à la liste
            results.append([periode, mse, mae])

        # Afficher les résultats dans le terminal
        headers = ["Period", "MSE", "MAE"]
        print("\n")
        print("Tableau des erreurs")
        print(tabulate(results, headers=headers, floatfmt=".4f"))
        print("\n")

    def run_modelization(self):
        self.initialize_model()
        self.fit_model()
        self.get_summary()
        self.plot_recession_probabilty()
        self.get_metrics()
        self.get_transition_matrix()

