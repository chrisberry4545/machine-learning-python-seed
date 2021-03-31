import numpy as np
from matplotlib import pyplot

def plot_model(prediction_df, feature, label):

  pyplot.xlabel(feature)
  pyplot.ylabel(label)

  pyplot.scatter(prediction_df[feature], prediction_df[label], alpha=0.1)

  features = {name:np.array(value) for name, value in prediction_df.items()}
  features.pop(label)
  pyplot.scatter(features[feature], prediction_df[label], c='r')

  pyplot.show()
