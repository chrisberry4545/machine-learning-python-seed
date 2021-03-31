import pandas as pd
import numpy as np

def get_data(data_url):
  base_data_frame = pd.read_csv(data_url)
  training_df = base_data_frame.sample(frac=0.8, random_state=200)
  test_df = base_data_frame.drop(training_df.index)
  return training_df, test_df
