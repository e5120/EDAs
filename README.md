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

## How to install
1. `git clone https://github.com/e5120/EDAs.git`
2. `cd EDAs`
3. `pip install -r requirements.txt`
4. `pip install -e .`

## Usage
1. `cd scripts`
2. Rewrite a script file `xxx.sh`, if necessary. See output of `python ../main.py -h` for details of each parameter.
3. Execute a command `bash xxx.sh`.

## Optimization Methods
- [PBIL](https://apps.dtic.mil/docs/citations/ADA282654)
- [UMDA](http://www.muehlenbein.org/estbin96.pdf)
- [CGA](https://ieeexplore.ieee.org/document/797971)
- [MIMIC](https://papers.nips.cc/paper/1328-mimic-finding-optima-by-estimating-probability-densities.pdf)
- [ECGA](https://www.researchgate.net/publication/2460502_Linkage_Learning_via_Probabilistic_Modeling_in_the_ECGA)
- [AffEDA](https://ieeexplore.ieee.org/document/6793952)
- [BOA](https://dl.acm.org/doi/pdf/10.5555/2933923.2933973)

## Objective function
The search space is *D*-dimensional bit-strings **c** &in; {0,1}<sup>D</sup>.
- OneMax: ![\begin{equation*}
f(\boldsymbol{c})=\sum_{i=1}^{D}c_i
\end{equation*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Bequation%2A%7D%0Af%28%5Cboldsymbol%7Bc%7D%29%3D%5Csum_%7Bi%3D1%7D%5E%7BD%7Dc_i%0A%5Cend%7Bequation%2A%7D%0A)
- TwoMin: $f(\bm{c}, \bm{y})=\min(\sum_{i=1}^D|c_i-y_i|,\sum_{i=1}^D|(1-c_i)-y_i|)$
  - $\bm{y}$ is *D*-dimensional bit-strings which is generated randomly in advance.
- Four-peaks: $f(\bm{c})=\max(o(\bm{c}),z(\bm{c}))+$REWARD
  - $o(\bm{c})$ is the number of contiguous ones starting in Position 1.
  - $z(\bm{c})$ is the number of contiguous zeros ending in Position *D*.
  - if $o(\bm{c}) > T$ and $z(\bm{c}) > T$, then REWARD is *D*, else REWARD is 0.
    - $T$ is a user parameter.
- Deceptive-k Trap
- NK-landscape
- W-Model
