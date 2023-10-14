import logging

# Create logger
logger = logging.getLogger(__name__)

# Set default logging level
logger.setLevel(logging.INFO)

# Create and set formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create and add stream handler (console output)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
