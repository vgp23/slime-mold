# Run simulation

There are currently three major ways to run simulations: with an interactive GUI, without GUI but showing the scenes afterwards, and running multiple repeated experiments in parallel varying parameter values. All these simulations are run by using `cd src; python main.py`. At the end of `main.py` one can see the various options with comments explaining them. For running a single simulation the interactive GUI environment is most useful. In the GUI there are numerous shortcuts which are all listed below.

- spacebar: pause or unpause
- q or esc: quit simulation
- h: toggle agent history
- a: toggle agents
- t: toggle trail
- c: toggle food/chemo
- f: toggle food sources
- w: toggle walls
- g: toggle graph

# Run analysis

When you ran experiments all the results are saved in the `../results` folder. Be aware that this folder can be quite significant in size, as the final scene for all simulations are saved (e.g. for 6 parameters varied by 6 values this totals to 5.6Gb). Using `cd src; python analysis.py` all the results can be analysed. All plots for the report are created using this script. Again, at the end of the file one can find example snippets for analysing the results.