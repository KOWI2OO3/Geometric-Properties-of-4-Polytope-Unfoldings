
import generation as gen
import numpy as np

with open("output/facet_count.txt", "w") as results:
    for i in range(5, 41):
        hull = gen.generate_uniform_unit_convex(i)
        results.write("Points: " + str(i) + " -> " + str(len(hull.simplices)) + " Facets\n")