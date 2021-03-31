import numpy as np

def get_prediction(model, test_df):
  features_to_predict = {name:np.array(value) for name, value in test_df.items()}
  random_trained_examples = model.predict(features_to_predict)
  expanded_results = np.array(random_trained_examples)
  predictions = expanded_results[:,0]
  return predictions
