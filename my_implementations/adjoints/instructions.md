Under this 'Codex Code' heading, please build code for a trainable neural PDE. 

- First, you'll have to get true Allen Cahn (1d ) data; use a useful_functions.jl function for that. Please train on a single allen cahn trajectory starting with the sin initial condition, \kappa = 0.1. 
- build a neural pde whose objective is the L2 error between true and simulated trajectory. The network should be a simple sigmoid activated network, as described. 
-  you'll have to build a neural 'PDE' stepper that generates a forwards trajectory (this shouldn't be too hard). 
- build a function that calculates the gradient of the loss with respect to the parameters. This should use the adjiont method described in this and the other adjoints notebook! I should be able to read your comments or markdown cells and figure out which step of the process is happening when.
- you'll need something to actually update parameters according to the gradient. 
- document your training well, in a bunch of successive cells. I don't want it all in one function call, but a cell for each segment of the training with markdowns explaining what's going on; by segment, i mean the following: 
- when you do the training, start by training on single-timestep trajectories. Later, train on longer trajectories. Finally, train on the whole trajectory (if needed).
- for each segment of training, store the errors over time so i can plot them later. Also, once the trajectories get long enough to visualize, for each segment, store a trajectory (true and learned) so i can visualize them later. 


Guidelines
- do not be verbose! I've found that codex code tends to be way too many lines. It should be absolutely minimal. This means no error handling, no extra helper functions, etc.
- Be readable. If steps happen in an obvious order, the functions for those steps should be written and called in that order. Don't have some function like validate_results!(...) that is called at the end of some large set of helpers and then does all of the real work.s
- Don't call on large libraries. I want it to be in front of me, although that will be slow. 
- comment your code well so that everything is nicely explained
- document your work in my_implementations/adjoints/steps.md

