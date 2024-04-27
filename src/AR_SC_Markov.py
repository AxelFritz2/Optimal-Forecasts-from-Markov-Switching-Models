import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

class AR_SC_Markov():
    def __init__(self, df, n_states, order):
        self.df = df
        self.n_states = n_states
        self.order = order

    def initialize_model(self):
        print("Model Initialization : \n"
              "Model : Markov Auto Regression \n"
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
        df_temp = self.df.iloc[self.order:]

        # Plot the series and the smoothed probabilities
        fig, ax1 = plt.subplots()

        ax1.plot(df_temp.index, df_temp['GR_GNP'], label='GR_GNP')
        ax1.set_ylabel('GR_GNP', color='C0')
        ax1.tick_params(axis='y', labelcolor='C0')

        ax2 = ax1.twinx()
        ax2.fill_between(df_temp.index, smoothed_marginal_probabilities[1], step='pre', alpha=0.4, color='C1')
        ax2.set_ylabel('Probability of Regime 1', color='C1')
        ax2.tick_params(axis='y', labelcolor='C1')
        ax2.set_ylim([0, 1])

        fig.tight_layout()
        plt.show()

    def get_transition_matrix(self):
        p00, p10, const0, const1, sigma0, sigma1, ar1_L1, ar1_L2 = self.model_fitted.params

        log_odds = np.array([const0, const1])
        diff_log_odds = np.array(
            [log_odds[i] - log_odds[j] for i in range(len(log_odds)) for j in range(len(log_odds))]
        )

        transition_probs = np.exp(diff_log_odds) / (1 + np.exp(diff_log_odds))

        transition_matrix = np.array(
            [transition_probs[i:i + len(log_odds)] for i in range(0, len(transition_probs), len(log_odds))]
        )

        transition_matrix /= np.sum(transition_matrix, axis=1)[:, None]

        print("Transition Matrix : ")
        print(transition_matrix)

    def run_modelization(self):
        self.initialize_model()
        self.fit_model()
        self.get_summary()
        self.plot_in_sample_forecast()
        self.plot_recession_probabilty()
        self.get_transition_matrix()
