from datetime import datetime

import generation as gen
import numpy as np
from net import Net, constructAdjacencyGraph

# Experiment 1
# n_samples = 1_000
# m_instances = 5

# Experiment 2
# n_samples = 1_000
# m_instances = 10

# Experiment 2 V2
# n_samples = 10_000
# m_instances = 10

# Experiment 3
# n_samples = 1_000
# m_instances = 10

# Experiment 3 V2
# n_samples = 10_000
# m_instances = 10

# Debug
# n_samples = 1
# m_instances = 10

log_to_console = True
debug_logging = False

def log(logger, context):
    now = datetime.now()
    line = "[" + now.strftime('%H:%M:%S') + "." + str(int(now.microsecond / 1_000)) + "]: " + context + "\n"
    logger.write(line)
    if log_to_console:
        print(line)

def run_pipeline(min, max, step: float, func, m_instances: int, n_samples: int):
    startTime = datetime.now()
    with open("output/log.txt", "w") as logger:
        with open("output/results.txt", "w") as results:
            log(logger, "Starting polytopes computation")
            
            # for i in range(min, max + step, step):
            count = 0
            while True:
                i : float = min + step * count
                count += 1

                log(logger, "Starting Computation for polytopes with " + str(i))
                for m in range(m_instances):
                    log(logger, "Computating polytopes with " + str(i) + " #" + str(m + 1))
                    hull = func(i)
                    success_rate = pipeline_sample(hull, logger, n_samples)
                    results.write(str(i) + ", " + str(success_rate))
                    results.write(",\n")
                log(logger, "Finished polytopes with " + str(i))

                if i >= max:
                    break
            
        log(logger, "Computation finished! took " + str(datetime.now() - startTime))

def pipeline_sample(hull, logger, n_samples: int):
    graph, valid_indices = constructAdjacencyGraph(hull)
    
    n_success: int = 0
    for i in range(n_samples):
        if debug_logging:
            log(logger, "Starting sample number " + str(i))
        
        net = Net.sampleRandomSpanningTree(hull, graph, valid_indices)
        if net.is_unfolding_overlap_free():
            n_success += 1

    if n_success == 0:
        log(logger, "FOUND POLYTOPE WITHOUT UNFOLDINGS")

    return float(n_success) / float(n_samples)

# Experiment 1
# run_pipeline(5, 30, 1, lambda x: gen.generate_clustering_unit_convex(x, 0), 5, 1000)

# Experiment 2 ~ 16 hours
# run_pipeline(0, 1, 0.05, lambda x: gen.generate_clustering_unit_convex(12, x), 10, 10_000)

# Experiment 3 ~ 16 hours
# run_pipeline(1, 20, 1, lambda x: gen.generate_convex_with_deformation(10, np.array([x, 1, 1, 1])), 10, 10_000)