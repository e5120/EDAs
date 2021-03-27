# EDAs
Implementation of Estimation of Distribution Algorithms (EDAs).

EDA is a framework to optimize black-box discrete optimization problems.  
The algorithm of EDA is as follows.
1. Initialize a population *P* whose size is &lambda;.
2. Construct a population *S* which includes promising solutions in *P*.
3. Build an explicit probabilistic model *M* based on *S*.
4. Generate new &lambda;<sub>candidate</sub> solutions from *M* to construct a population *O*.
5. The solutions in *P* is replaced with those of *O*.
6. If termination conditions are met, then the algorithm is terminated, else go to (2).

## Setup
EDAs requires:
- Python >= 3.6

Install `EDAs` from the sources:
```
git clone https://github.com/e5120/EDAs.git
cd EDAs
pip install -r requirements.txt
pip install -e .
```

(Optional) If you want to use `main.py` or `eda/builder.py`, you need to install [BB-DOB](https://github.com/e5120/BB-DOB) project.

## Features
- [PBIL](https://apps.dtic.mil/docs/citations/ADA282654)
- [UMDA](http://www.muehlenbein.org/estbin96.pdf)
- [CGA](https://ieeexplore.ieee.org/document/797971)
- [MIMIC](https://papers.nips.cc/paper/1328-mimic-finding-optima-by-estimating-probability-densities.pdf)
- [ECGA](https://www.researchgate.net/publication/2460502_Linkage_Learning_via_Probabilistic_Modeling_in_the_ECGA)
- [AffEDA](https://ieeexplore.ieee.org/document/6793952)
- [BOA](https://dl.acm.org/doi/pdf/10.5555/2933923.2933973)

## Usage
1. `cd scripts`
2. Rewrite a script file `xxx.sh`, if necessary. See output of `python ../main.py -h` for details of each parameter.
3. Execute a command `bash xxx.sh`.
