from datetime import datetime

import generation as gen
import numpy as np

def log(logger, context):
    now = datetime.now()
    line = "[" + now.strftime('%H:%M:%S') + "." + str(int(now.microsecond / 1_000)) + "]: " + context + "\n"
    logger.write(line)

startTime = datetime.now()
with open("/output/log.txt", "w") as logger:
    with open("/output/results.txt", "w") as results:
        log(logger, "Computing polytopes")
        
    log(logger, "Computation finished! took " + str(datetime.now() - startTime))