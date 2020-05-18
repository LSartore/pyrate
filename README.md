# PyR@TE 3 (beta-version)

### New in v3.0:

- The core of the program is based on <a href="https://arxiv.org/abs/1906.04625">1906.04625</a>, where the general gauge couplings RGEs were computed up to 3 loops.
- Python 3 compatibility
- Drastic improvement of the performances (computation times of several hours reduced to some minutes)
- The structure of the model file has been rethought, and in particular the implementation of the Lagrangian. New features are available.
- Major improvements to the PyLie group-theoretical module (better performances + new functionalities)
- Many other small features


### Dependencies :

- Python &ge; 3.6
- PyYAML &ge; 5.3
- Sympy &ge; 1.5
- h5py &ge; 2.10
- Numpy &ge; 1.18
- Scipy &ge; 1.4
- Matplotlib &ge; 3.1


### Download:

The only thing to do is to clone this repository, and begin working in PyR@TE 3's main folder.  

### Description:

PyR@TE is a Python code that computes the renormalization group equations (RGEs) for any renormalizable non-supersymmetric model. After the gauge groups, the particle content and the Lagrangian of have been defined in a model file, PyR@TE calculates the RGEs for all of the couplings at the one- or two-loop level, and up to the three-loop level for gauge couplings.  

### How to run PyR@TE

While there is no official documentation available yet for the version 3, please refer yourself to the example notebooks in `doc/` :
- `example.ipynb` explains some of the general features of PyR@TE 3 and shows how to use it ;
- `CGCs.ipynb` explains how to interact with PyLie's database and to use Clebsch-Gordan coefficients (CGCs) when building a Lagrangian  

Please note that these two notebooks do not cover the entirety of PyR@TE 3's functionalities, but may still serve as a useful starting point. A comprehensive documentation will be released in the future.



For suggestions, bug reports or any comments please contact the author at : 

sartore at lpsc.in2p3.fr

