import numpy as np
from matplotlib import pyplot

def plot_model(predictions, feature, label, test_df):

  pyplot.xlabel(feature)
  pyplot.ylabel(label)

  pyplot.scatter(test_df[feature], test_df[label], alpha=0.1)

  test_features = {name:np.array(value) for name, value in test_df.items()}
  test_features.pop(label)
  pyplot.scatter(test_features[feature], predictions, c='r')

  pyplot.show()

  return predictions
