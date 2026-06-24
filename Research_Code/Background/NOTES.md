# Some thoughts I'm having, 6/12/2026

## AC ROMs 2D

### Experiment 1

Questions:
- Which is more important, POD or DEIM modes? Is it just about singular value capture?
  - it seems like with just 1 DEIM mode, we're good, but we need more than 1 POD mode. Is this specific to this initial condition?
  - I'm doing the I_1, trace-style singular value capture; would it be better to do the I_2, squared, `energy-style' singular value capture? 
- As always, what's a good error metric? should it be relative? gradient L2 or just L2? Energy?
  - energy seems to depend a lot on POD modes...


### Experiment 2

Questions
- it looks like FOM complexity is about N^3; is this expected?
  -  how does this compare to diffusion?
  -  is this integrator dependent?
-  without projection, our speedups seem to be around N^3; with projection, they seem to be about N^2
   - again, how does this compare to diffusion?
   - again, is this integrator dependent?

### Experiment 3

Questions
- it seems basically impossible to transfer from $\kappa = 0.01$ to $=0.001$; why? Judging by relative errors. It also doesn't seem like adding more modes is the issue. 
  - is this true for diffusion? **NO! It's Not!** Diffusion can learn much better
    - is this because the trajectory changes so drastically? 


# To Do

- project down from fom then back up for best possible result 
- L2 or some other solution based norm POD basis vs trajectory