import numpy as np

def get_dataframe_prediction(model, dataframe):
  data_frame_array = {name:np.array(value) for name, value in dataframe.items()}
  prediction = get_prediction(my_model, data_frame_array);
  return prediction
