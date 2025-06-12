import time
import logging

def log_progress(message, level="info"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    
    if level == "info":
        logging.info(log_msg)
    elif level == "warning":
        logging.warning(log_msg)
    else:
        logging.error(log_msg)
        
    print(log_msg)

def setup_logger(log_file="pipeline.log"):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="[%(asctime)s] %(message)s"
    )
