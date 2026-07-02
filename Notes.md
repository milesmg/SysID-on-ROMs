# Some Research Questions

# Week 4

### `miles_differentiable_neural_PDEs.ipynb`

- Training on FOM
  - why do we train on a whole trajectory if we know at the beginning we're diverging super fast? wouldn't it make sense to start by training on a bunch of short steps?
  - We're learning $N_{\theta}: [-1,1] \to \mathbb R$. Is there a way to tell which points in  $[-1,1]$ are most represented in our dataset vs which ones are learned best?
    -  Because as Dr. Urban was saying there are only 3 points in the true trajectory that are at u = 0, so you would expect this to be undersampled, but it seems like the NN learned function does *best* there. 
    - it seems like we do worst at the ends, which isn't surpising, and at the humps, which isn't surprising; those are the places we'd avoid in a sim...




- debugging compiled autojacvec
  - Enzyme didn't work (yet)
  - ReverseDiffVJP(false)  (this is without compilation)
    - worked with 40 steps
  - ReverseDiffVJP(true) (ie with compilation)
    - worked! with 400 steps
  - EnzymeVJP (with compilation)
    - always seems to fail around 60-180 with 400 steps 
  - If it works with ReverseDiff (as opposed to Enzyme) does it matter? Should I just use Reverse?
  - what was the issue with training the NN?
    - got Chat to write a big debug. The issue was a native julia propcess crash (as opposed to something running super slow.)
      - After a given iteration's ODE solve, the Julia notebook process disappears
      - macOS makes a crash report with a GC crash?
      - Chat: 'This indicates memory corruption occurred before the garbage collector encountered the invalid memory.'
    - Chat thinks it's Enzyme:
      - uses native LLVM-generated code 
        - ie it analyzes pre-compiled code and generates its own compilations based on that ? 
          - hence no Julia errors like a MethodError or stack trace
      - its libraries occur in the crash report
        - though the visible failure is inside Julia's GC
    - i can switch to a different autojacvec to test; I can also turn off compile
      - GaussAdjoint(autojacvec=ReverseDiffVJP(compile=false))
      - Safer but SLOWER!
        - according to SciML documentation, only compile if the RHS has no data-dependent branching (eg if, while, etc. )
          - mine doesn't?

- Enzyme vs ReverseDiff(true):
  - ReverseDiffVJP (true) appears to be slightly higher level than Enzyme 
  - Could be a version compatibility issue? should update?
    - okay, I updated, and now when I run 
        `Optimization.solve(optprob,OptimizationOptimisers.Adam(η, β);maxiters=1, )` with Enzyme the julia notebook crashes again. So I'm abandoning Enzyme 

- DeBugging ROM Training
  - going to switch to ReverseDiffVJP(false) for now
    - that worked (for 400 iterations); now going to try ReverseDiffVJP(true)

### General Questions

- Conceptual questions about SysID on ROM
  - So the idea is take a trajectory with f(u) snapshots to build POD / DEIM modes
    - but then are you really doing SysID? You already have f snapshots...
    - could you do this without f snapshots?
      - you'd need to pick DEIM points somehow...
      - you could do subtraction to get the nonlinearity (ie $u_t - Au$) but if you don't have exact timestepping data you can't really do that anyway
      - you could learn $B$, the interpolation matrix, and $F$, simulataneously
        - could try to choose the DEIM points based on spatial basis?
      - could also just learn the nonlinear update directly, but then can you do real sys ID?
  - why is our error on the full trajectory instead of reduced? because the minimum RMSE is the error of our true DEIM operator anyway (b/c generated from SVD), so we're just adding a constant floor error
  - Is there a nice way to learn the low dimensional representation update rule, but get out a high dimensional **pointwise** function?


- Building a non-DEIM interpolator
  - do we even want to train on the ROM? If we're learning per NN evaluation, then does it matter?
    -  more evaluations per time step in the FOM might be fine if we're learning per eval, especially if the NN is the main cost
    - if the NN is the main cost, then the only reason not to do FOM is because we're sampling a more 'learnable' set of points / trajectory by learning on the ROM

### `training_on_FOM.ipynb`

- Building a .jl file with helper functions
  - so I got an error because I wasn't passing in properly destructured Lux NN parameters, and apparently this breaks ReverseDiffVJP(compile = true)
    - apparently i need to use componentvec storage of parameters instead of named tuples
    - it wants me to switch to Enzyme if i'm not going to destructure
      - so I'm going to try not destructuring and just using Enzyme, but I'm expecting that this won't work and I'll have to use ReverseDiff and de/re structure. 

### `training_on_ROM.ipynb`


- Hyperparameter optimization
  - Optimization Params: η = 0.05; β = (0.9, 0.99); N_iter = 400
    - Big jump in error around 300 iterates but falls back down around 400
  - costs a ton to increase N_obs (number of observation times); why? what should this be? 
    - we appear to be O(N_obs) here 
    - 
  - learn rate scheduling? Adam params? N_iterates? 
  - number of modes?



### Cluster Compatibility

- What do I want? Well, as a first pass, I want to be able to run code exactly like my jupyter notebooks, and just have them run way faster because they're on a GPU. That would be cool. In this case, I can put in parameters and save directories manually. 
- Then I'll want to do timing tests, I guess. 
- Eventually, for hyperparameter optimization, I'll want to run a ton of different optimizations; effectively I want to do a sweep. In this case, I'll want to be able to pass in parameters via an external script, which might make it nice to be able to adjust them via the command line
  - but do i really need this mutable argument structure? because it's a mess to code. If I'm writing a script, I can definitely pass arguments in a fixed, required format. 

## Week 5

### Building Julia / package stuff into repo so it's easier to update and for Chat to work with

- Found: 
  - R=20, J4/B1 (julia / BLAS threads): 226.29 s / 100 iters = 2.26 s/iter 
    - projected 48,000 iterations: ~30.2 hours
    - Do we really need this many? 
  - R=40, J4/B1: 321.86 s / 100 iters = 3.22 s/iter
  - Seems like J8/B1 is the best for this specific paramerization; what does this depend on ? How do i choose? Idk.
    - maybe Julia can do this automatically
- Built notebooks to visualize data 
