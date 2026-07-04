# General Principles for Agentic AI Work

- Minimality:
  - Write as little code as possible
  - Do not default to error handling. Only add extra error catches when requested.

- Testing and compatibility:
  -  Make sure that code will run as expected
  -  Changes that are made to FOM code should likely be made to ROM code as well; please ask if this is not specified

- Commenting
  - wherever you make edits, add a one line comment above describing what you've changed
  - this comment should begin with ### ADJUSTED: (description)
  - document your edit in a new .md file in the codex_logs folder. 
    - This description should be short, unless large scale changes were made. 
    - This file should also contain a list of files edited, with edits made to each

- Legibility 
  - code should be written so that it is readable
  - functions that are meant to run in order should be written in that order, and should be called sequentially, rather than in a nested / dependent manner
  - whenever changes are made to a function or class, the relevant docstring should be updated. This update should be minimal. 

- Working with HPC code / data
  - when writing code for the HPC, you won't have access to the files on the cluster
  - be cognizant of when you don't have access to a file. Do not assume, or try to work around lack of a necessary file; rather, say that you are missing something, and give instructions for how it can be provided to you
  - Important: Do not rsync, or write an rsync command, unless you're absolutely sure that it won't delete any important files (whether you're going from HPC to local or vice versa)
  - When writing rsync commands for this repo, use `--filter='dir-merge .rsync-filter'`, exclude `.git/`, and use `--delete` only when explicitly requested so per-directory `.rsync-filter` files protect machine-built/local-only paths like `Julia/depot/`.

- Questions: Ask clarifying questions rather than proceeding if you're unsure !s

- Github: Don't commit or push anything. 