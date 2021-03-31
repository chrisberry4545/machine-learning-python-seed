from create_model import create_model
from train_model import train_model
from plot_loss_curve import plot_loss_curve
from plot_model import plot_model
import numpy as np
from get_data import get_data
from evaluate_results import evaluate_results
import pandas as pd
from get_prediction import get_prediction

def generate_model(
  label_name,
  main_feature,
  columns_to_include,
  learning_rate,
  epochs,
  batch_size,
  neural_network_structure,
  dropout_rate,
  output_json_results,
  output_loss_curve,
  output_results_plot
):
  training_df, test_df, scaler = get_data()
  columns_to_remove = []
  # Remove other columns
  for col in training_df.columns:
    if col not in columns_to_include:
      columns_to_remove.append(col)
      del training_df[col]

  feature_column_names = columns_to_include.copy()
  feature_column_names.remove(label_name)

  # Create and compile the model's topography.
  model = create_model(learning_rate, feature_column_names, neural_network_structure, dropout_rate)

  # Train the model on the training set.
  epochs, rmse = train_model(model, training_df, epochs, batch_size, label_name)

  if output_loss_curve:
    plot_loss_curve(epochs, rmse)

  predictions = get_prediction(model, test_df)

  normalized_predictions_df = test_df.copy()
  normalized_predictions_df[label_name] = predictions

  de_normalized_input_features = scaler.inverse_transform(test_df)
  de_normalized_input = pd.DataFrame(de_normalized_input_features, columns=test_df.columns)

  de_normalized_predictions_features = scaler.inverse_transform(normalized_predictions_df)
  de_normalized_predictions = pd.DataFrame(de_normalized_predictions_features, columns=normalized_predictions_df.columns)

  de_normalized_main_feature_array = de_normalized_input[main_feature].to_numpy()
  de_normalized_real_values_array = de_normalized_input[label_name].to_numpy()
  de_normalized_predictions_array = de_normalized_predictions[label_name].to_numpy()

  if output_results_plot:
    plot_model(de_normalized_predictions, main_feature, label_name)

  test_features = {name:np.array(value) for name, value in test_df.items()}
  label_res = test_features.pop(label_name)
  evaluation = model.evaluate(test_features, label_res)
  print('model evaluation:', evaluation)

  evaluate_results(de_normalized_main_feature_array, de_normalized_predictions_array, de_normalized_real_values_array, output_json_results)
