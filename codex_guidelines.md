# General Principles for Agentic AI Work

- Minimality:
  - Write as little code as possible
  - Do not default to error handling. Only add extra error catches when requested.

- Testing: Make sure that code will run as expected

- Commenting
  - wherever you make edits, add a one line comment above describing what you've changed
  - this comment should begin with ### ADJUSTED: (description)
  - document your edit in a new .md file in the codex_logs folder. This description should be very short, unless large scale changes were made. 

- Legibility 
  - code should be written so that it is readable
  - functions that are meant to run in order should be written in that order, and should be called sequentially, rather than in a nested / dependent manner
  - whenever changes are made to a function or class, the relevant docstring should be updated. This update should be minimal. 
