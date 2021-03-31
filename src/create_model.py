import tensorflow as tf
from tensorflow.keras import layers

def create_model(my_learning_rate, feature_column_names, neural_network_structure, dropout_rate = 0.2):
  feature_columns = []
  for feature_column_name in feature_column_names:
    feature = tf.feature_column.numeric_column(feature_column_name)
    feature_columns.append(feature)

  feature_layer = layers.DenseFeatures(feature_columns)

  model = tf.keras.models.Sequential()
  model.add(feature_layer)

  index = 0
  for number_of_nodes in neural_network_structure:
    index = index + 1
    if index == 2 and dropout_rate > 0:
      print('adding dropout layer', dropout_rate)
      model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    layer_name = 'Hidden' + str(index)
    print('adding hidden layer', layer_name, 'with', number_of_nodes, 'nodes')
    model.add(tf.keras.layers.Dense(units=number_of_nodes,
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                    name=layer_name))

  model.add(tf.keras.layers.Dense(units=1,
                                  name='Output'))

  model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

  return model
