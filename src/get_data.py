import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_data(data_url = './data/california_housing.csv'):
  base_data_frame = pd.read_csv(data_url)

  values = base_data_frame.values
  scaler = StandardScaler()
  scaler.fit_transform(values)
  standardized = scaler.transform(base_data_frame)
  base_data_frame_norm = pd.DataFrame(standardized, columns=base_data_frame.columns)

  training_df = base_data_frame_norm.sample(frac=0.8, random_state=200)
  test_df = base_data_frame_norm.drop(training_df.index)

  return training_df, test_df, scaler
