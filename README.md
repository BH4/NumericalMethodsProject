# NumericalMethodsProject
Numerically time evolve a wavefunction within a smooth, 1d potential

This is a project I made for a numerical methods class. I was tasked with evolving an initial wavefunction in a quartic potential. I chose to do this by finding the overlap of the initial function with the eigenstates of the quartic potential then time evolving the eigenstates. Analytically this is the most simple technique for time evolution but computationally this requires very high accuracy.

I tried to make this code a general as possible so it should be reasonable to switch the Hamiltonian out with any Hamiltonian satisfying the same boundary conditions. This requires changing the coordinates to the same ones I have used however.

Since I completely stopped work on this right after creating a presentation on it, it is a little bit messy. Some functions are defined specifically to create a graph or movie I used in the presentation and there are some comments which are just old code or code I would switch back and forth to in order to test my code.