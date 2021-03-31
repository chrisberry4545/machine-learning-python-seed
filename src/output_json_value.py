import json

def output_json_value(data_frame, column_name):
  return json.dumps(data_frame[column_name].tolist())
