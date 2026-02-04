import logging
from datetime import datetime
import os
LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y-%H:%M:%S')}.log"

LOG_PATH = os.path.join(os.getcwd(),'log')

os.makedirs(LOG_PATH, exist_ok = True)

LOG_FILE_PATH = os.path.join(LOG_PATH,LOG_FILE_NAME)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

