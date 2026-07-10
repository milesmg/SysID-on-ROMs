### ADJUSTED: Protect virtual queue logs during src rsyncs.

Files edited:
- `Research_Code/src/HPC/Tools/Sweeps/.rsync-filter`: excludes `virtual_sweep_queue.log` and `virtual_sweep_queue_queue.tsv` so src syncs do not overwrite machine-local virtual queue state logs.
