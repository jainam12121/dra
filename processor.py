import joblib
import numpy as np

# Load the model
loaded_model = joblib.load('model.joblib')  # Use joblib to load the model

def pre_process(input_data):
    """Prepares the input data for prediction."""
    input_data_as_numpy_array = np.asarray(input_data)
    input_reshape = input_data_as_numpy_array.reshape(1, -1)
    return input_reshape

def post_process(prediction):
    """Interprets the prediction result."""
    if prediction[0] == 0:
        result = 'The person is not diabetic'
    else:
        result = 'The person is diabetic'
    return result

# Example input
input_data = (137, 138, 43, 33)  # Example input

# Pre-process the input
processed_input = pre_process(input_data)

# Make prediction
prediction = loaded_model.predict(processed_input)

# Post-process the prediction
result = post_process(prediction)
print(result)
