This folder contains the SPECFEM input files to simulate the example case we're comparing the extended reflectivity model to. These files are all you need to run the same example yourself (and the SPECFEM3D Cartesian code obviously).

The problem geometry is a rectange with five layers, and the mesh is generated with SPECFEM's internal mesher xmeshfem3D.

The results folder "OUTPUT_FILES" includes some information on the run (in case you're interested but can't run the model yourself now).

If you want to run this example yourself, just make sure you have SPECFEM3D Cartesian compiled and that the symbolic links for the executables in run_this_example.sh are pointing to the right folder. You may want/need to change the number of processors in the mesher and solver input files to match your system (currently the input files specify 20 processors). On a 20 processor system the simulation took 4 hours and 53 minutes.