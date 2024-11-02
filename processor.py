import joblib
import numpy as np
from vipas.exceptions import UnauthorizedException, NotFoundException, BadRequestException, ForbiddenException, ConnectionException, ClientException
from vipas.logger import LoggerClient

# Initialize the logger
logger = LoggerClient(__name__)

# Load the model
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
        # If input_data is a string, convert it to a tuple of numbers
        if isinstance(input_data, str):
            input_data = eval(input_data)  # Evaluate string to get tuple

        # Convert to numpy array with float type for numeric compatibility
        input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)
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

# Example input
# input_data = (137, 138, 43, 33)  # Example input

# # Pre-process the input
# try:
#     processed_input = pre_process(input_data)

#     # Make prediction
#     prediction = loaded_model.predict(processed_input)

#     # Post-process the prediction
#     result = post_process(prediction)
#     print(result)
# except Exception as e:
#     logger.critical(f"An error occurred during processing: {str(e)}")
