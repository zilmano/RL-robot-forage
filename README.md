# Reinforcement Learning For Continuous Search Tasks With Unknown Priors

Please see the project description in the following paper:
https://drive.google.com/file/d/1DIYh0xE6ExES_ugUo415CrLveOcnwkmn/view?usp=sharing

The main executbale file is main.py.

The pythondependecies needed to run this project are:
numpy, tqdm, matplotlib, scipy, enum

To run:

**python main.py**


- By default, it will run the dynaQ algorithm on a 8x8 grid, for ten experiments, and then it will print out the result plots to the screen,
  and save them to the files "results.png" and "learning_curve.png". Expected run time - about one hour.

- If you which to ran the approximation algorithm, uncomment the very last line of main.py, #runApproximation().
  It takes some time long time to run.

- If you wish to change the grid size from 8x8, please change 'n' and 'm' on lines 377 and 378 of main.py (right at the start of the
  main function). 
  The biggest grid we were able to run was 15x15, for anything bigger our machine ran out of memory when python tried to 
  allocate the tranistion matrix table (it is pretty big, size of about 13500*13500*4 for a 15x15 grid, with float64 entries)
  For 15x15 grid, expect several hours of running.
  
 
