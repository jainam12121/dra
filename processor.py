import joblib
import numpy as np
from vipas.exceptions import UnauthorizedException, NotFoundException, BadRequestException, ForbiddenException, ConnectionException, ClientException
from vipas.logger import LoggerClient

# Initialize the logger
logger = LoggerClient(__name__)

# # Load the model
# try:
#     loaded_model = joblib.load('model.joblib')
#     logger.info("Model loaded successfully.")
# except FileNotFoundError as err:
#     logger.critical(f"Model file not found: {err}")
#     raise
# except Exception as err:
#     logger.critical(f"Unexpected error while loading model: {str(err)}")
#     raise

def pre_process(input_data):
    """Prepares the input data for prediction with exception handling."""
    try:
        # Split the input string by commas and convert each part to float
        input_values = [float(value.strip()) for value in input_data.split(",")]

        # Convert the inputs directly to a numpy array with float type
        input_data_as_numpy_array = np.array(input_values, dtype=np.float64)

        # Reshape to ensure it's 2D and convert to list for JSON serialization
        input_reshape = input_data_as_numpy_array.reshape(1, -1).tolist()  
        
        logger.info("Preprocessing completed successfully.")
        return input_reshape  # Return as a list for JSON compatibility
    except ValueError as err:
        logger.error(f"ValueError during preprocessing: {err} - Ensure input data is numeric.")
        raise
    except ConnectionException as err:
        logger.error(f"ConnectionException during preprocessing: {err}")
        raise
    except ClientException as err:
        logger.error(f"ClientException during preprocessing: {err}")
        raise
    except Exception as err:
        logger.critical(f"Unexpected error during preprocessing: {str(err)}")
        raise

def post_process(prediction):
    """Interprets the prediction result with exception handling."""
    try:
        if prediction[0] == 0:
            result = 'The person is not diabetic'
        else:
            result = 'The person is diabetic'
        logger.info("Postprocessing completed successfully.")
        return result
    except ConnectionException as err:
        logger.error(f"ConnectionException during postprocessing: {err}")
        raise
    except ClientException as err:
        logger.error(f"ClientException during postprocessing: {err}")
        raise
    except Exception as err:
        logger.critical(f"Unexpected error during postprocessing: {str(err)}")
        raise

# Main execution with user input
# if __name__ == "__main__":
#     try:
#         # Prompt user for input
#         user_input = input("Enter values separated by commas (e.g., 137, 138, 43, 33): ")

#         # Pre-process the input values
#         processed_input = pre_process(user_input)

#         # Make a prediction
#         prediction = loaded_model.predict(processed_input)  # Now processed_input is correctly shaped

#         # Post-process the prediction and get the result
#         result = post_process(prediction)
#         print(result)
#     except Exception as e:
#         logger.critical(f"An error occurred during processing: {str(e)}")
