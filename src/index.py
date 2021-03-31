from generate_model import generate_model

# Model variables
learning_rate=0.001
epochs=10
batch_size=50
neural_network_structure=[12, 10]
dropout_rate=0
output_json_results=False
output_loss_curve=True
output_results_plot=True

label_name = 'median_house_value'
main_feature = 'median_income'
columns_to_include = [
                      label_name,
                      'longitude',
                      'latitude',
                      'housing_median_age',
                      'total_rooms',
                      'total_bedrooms',
                      'population',
                      'households',
                      'median_income',
            ]

generate_model(
    label_name=label_name,
    main_feature=main_feature,
    columns_to_include=columns_to_include,
    learning_rate=learning_rate,
    epochs=epochs,
    batch_size=batch_size,
    neural_network_structure=neural_network_structure,
    dropout_rate=dropout_rate,
    output_json_results=output_json_results,
    output_loss_curve=output_loss_curve,
    output_results_plot=output_results_plot
  )
