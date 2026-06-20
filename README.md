# 26 Bachelor Research Project 4D Unfolding
This is the code used for the experimental investigation of unfoldability of the surface of 4-polytopes.
Serving as my bachelor project at the TU Delft.

# Code structure
All code is split into seperate files to make finding sections of code easier. 

- `generation.py` handles the generation of convex hulls
- `geometry.py` handles geometry intersection (SAT)
- `net.py` handles geometry transformations in the unfolding process
- `pipeline.py` contains the main pipeline, also houses the selected experiment at the bottom of the file

## Switching Experiment
At the bottom of `pipeline.py` there are a couple lines for enabling a certain experiment
```
# Experiment 1
# run_pipeline(5, 30, 1, lambda x: gen.generate_clustering_unit_convex(x, 0), 5, 1000)

# Experiment 2
# run_pipeline(0, 1, 0.05, lambda x: gen.generate_clustering_unit_convex(12, x), 10, 10_000)

# Experiment 3
# run_pipeline(1, 20, 1, lambda x: gen.generate_convex_with_deformation(10, np.array([x, 1, 1, 1])), 10, 10_000)
```

These can be uncommented to select a single experiment to be run. 

*Note*: Make sure to only enable a single experiment as only the results of the last performed experiment will be saved.

## Extra
Additionally this project includes a couple extra files which are used during testing and development.

- `facet_count.py` a small script which outputs the facet count resutling from generating a convex of a certain amount of points (5-40)
- `visualize.py` used to visualize the geometric properties in 2D on a plot
- `main.py` is primarily used to test with small sections of the code

# Docker
The experiments were performed using docker containers such that it is easier to reproduce.

## Starting the experiment
Make sure to have the desired experiment uncommented in `src/pipeline.py` at the bottom. 

To start the experiment run in the root of the project:
```
docker compose up --build -d
```

When the container is stopped the experiment is done.

## Retrieving data
To retrieve the data from the container, make sure the container has stopped itself indicating the experiment being done.
then run in the root of the project:
```
docker cp polytope-experiment:/output ./results
```

Which copies the results from the conainer into a folder `results` in the root of the project.

# Cleaning up
To clean up all generated content from the experiment run these 3 commands
```
docker compose down
docker volume rm polytope_experiment_data
docker image rm polytope-experiment
```