from output_json_value import output_json_value
from format_percent_for_output import format_percent_for_output
import pandas as pd

def evaluate_results(feature_values, predictions, real_values, output_json_results=False):
  result_dictionary = {
    "feature_values": feature_values,
    "predictions": predictions,
    "real_values": real_values
  }
  result_frame = pd.DataFrame(result_dictionary)
  result_frame['difference'] = (result_frame['real_values'] - result_frame['predictions']).abs()
  result_frame['percent_difference'] = result_frame['difference'] / result_frame['real_values']
  print(result_frame.head(30))
  average_percentage_out = result_frame['percent_difference'].mean()
  amount_within_10_percent = (result_frame['percent_difference'] <= 0.1).sum()
  total = result_frame['percent_difference'].count()
  percent_within_10_percent = amount_within_10_percent / total

  print('Total:', total)
  print('Average percentage error:', format_percent_for_output(average_percentage_out))
  print('Percent of total within 10%:', format_percent_for_output(percent_within_10_percent))

  if output_json_results:
    print('feature_values', output_json_value(result_frame, 'feature_values'))
    print('predictions', output_json_value(result_frame, 'predictions'))
    print('real_values', output_json_value(result_frame, 'real_values'))
    print('percent_difference', output_json_value(result_frame, 'percent_difference'))
