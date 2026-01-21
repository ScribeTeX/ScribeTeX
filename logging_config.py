"""
ScribeTeX Logging Configuration
JSON-formatted logging for the application.
"""
import logging
import sys

from pythonjsonlogger import jsonlogger

JSON_FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s %(filename)s %(lineno)d'


def setup_logging(level: int = logging.INFO):
    """
    Configure JSON-formatted logging to stdout.
    
    Args:
        level: Logging level (default: INFO)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Avoid adding duplicate handlers
    if any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        return
    
    # Create JSON formatter
    log_handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(JSON_FORMAT)
    log_handler.setFormatter(formatter)
    
    root_logger.addHandler(log_handler)
