### ADJUSTED: Cap reference saved times at 500 across HPC and local tooling.

Files edited:
- `Research_Code/src/HPC/Tools/hpc_common.jl`: documented and commented the 500-reference-save cap in `build_ac_reference`.
- `Research_Code/src/Visualizations/optimization_visualizations.jl`: capped local visualization replay times at 500, including old metadata with large `reference_saved_times` or `reference_save_count`.
- `Research_Code/src/Visualizations/optimization_visualizations_2D.jl`: applied the same 500-time cap for 2D visualization tooling.
- `Research_Code/Optimization/Local/training_on_FOM.ipynb`: capped the manually constructed reference `saveat` grid at 500.
- `Research_Code/Optimization/Local/training_on_ROM.ipynb`: capped the manually constructed reference `saveat` grid at 500.
