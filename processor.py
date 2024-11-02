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
        input_reshape = input_data_as_numpy_array.reshape(1, -1)
        logger.info("Preprocessing completed successfully.")
        return input_reshape.tolist()  # Convert ndarray to list for JSON compatibility
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

def post_process(input_data):
    """Makes a prediction and interprets the result with exception handling."""
  
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

#         # Post-process the prediction and get the result
#         result = post_process(processed_input)
#         print(result)
#     except Exception as e:
#         logger.critical(f"An error occurred during processing: {str(e)}")
