# RL-stochastic-search


The main executbale file is main.py.
The dependecies needed to run this project are:

To run:

\bpython main.py

- By default, it will run the dynaQ algorithm on a 8x8 grid, for ten experiments, and then it will print out the result plots to the screen,
  and save them to the files "results.png" and "learning_curve.png". Expected run time - about one hour

- If you which to ran the approximation algorithm, uncomment the very last line of main.py, #runApproximation().
  It takes some time long time to run.

- If you wish to change the grid size from 8x8, please change 'n' and 'm' on lines 377 and 378 of main.py (right at the start of the
  main function). 
  The biggest grid we were able to run was 15x15, for anything bigger our machine ran out of memory when python tried to 
  allocate the tranistion matrix table (it is pretty big, size of about 13500*13500*4 for a 15x15 grid, with float64 entries)
  
 
