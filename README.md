# Computational Neuroscience report

[The project report](https://dl.dropboxusercontent.com/u/47395591/Uni%20Projects/Computational_Neuroscience_Project.pdf)

[Original paper](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0117465)

[Original MATLAB code](https://github.com/IsaacLab/HNA-robotic-model/)

This project was about implementing an experiment by Aguilera et al. in which they artificially evolve a simple simulated robot to move towards a light source. The robot is controlled by a biologically realistic neural network which produces oscillations, and can dynamically adapt its parameters.

I found this difficult, because the original code was disorganised and very difficult to read, with single letter variable names (like `zz` and `a`), and data encoded in different rows and columns of various matrices (sadly a convention MATLAB seems to encourage). Whilst their Methods section was well written and detailed, unfortunately the symbols used in the paper did not match up to the variables in the code.

The implementation was unsuccessful; the simulation ran without errors, but did not reproduce similar results. Whilst in my report I (perhaps unfairly) blame difficulties in understanding the original code, in hindsight I wish I had applied a more test driven approach. Whilst there are some [tests](https://github.com/mbryantlibrary/CNproj/tree/master/src/Tests), they only covered very limited functionality, and I do wonder how many more bugs might have been uncovered with a more rigorous approach. When beginning the project I was under a little time pressure and probably thought it'd save time not writing many tests, but in hindsight I've realised this is a false economy (and [have applied it in more recent projects](https://github.com/mbryantlibrary/ImplicitForwardModels)).
