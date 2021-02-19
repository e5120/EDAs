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
- OneMax: <img src="https://latex.codecogs.com/gif.latex?\inline&space;f(\boldsymbol{c})=\sum_{i=1}^Dc_i" />
- TwoMin: <img src="https://latex.codecogs.com/gif.latex?\inline&space;f(\boldsymbol{c},&space;\boldsymbol{y})=\min(\sum_{i=1}^D|c_i-y_i|,\sum_{i=1}^D|(1-c_i)-y_i|)" />
  
  - **y** is *D*-dimensional bit-strings which is generated randomly in advance.

- Four-peaks: <img src="https://latex.codecogs.com/gif.latex?\inline&space;f(\boldsymbol{c})=\max(o(\boldsymbol{c}),z(\boldsymbol{c}))&plus;\textrm{REWARD}" />

  - *o*(**c**) is the number of contiguous ones starting in Position 1.
  - *z*(**c**) is the number of contiguous zeros ending in Position *D*.
  - if *o*(**c**) > *T* and *z*(**c**) > *T*, then REWARD is *D*, else REWARD is 0.
    - *T* is a user parameter.
- Deceptive-k Trap: There is a user parameter *k* which determines the number of dependencies of variables.

  <img src="https://latex.codecogs.com/gif.latex?f(\boldsymbol{c})&space;=&space;\sum_{i=0}^{D/3-1}g(c_{3i&plus;1},c_{3i&plus;2},c_{3i&plus;3})," />
  <br>
  <img src="https://latex.codecogs.com/gif.latex?g(c_1,&space;c_2,&space;c_3)&space;=&space;\left\{&space;\begin{array}{ll}&space;1-d&space;&&space;\sum_{i}c_i&space;=&space;0&space;\\&space;1-2d&space;&&space;\sum_{i}c_i&space;=&space;1&space;\\&space;0&space;&&space;\sum_{i}c_i&space;=&space;2&space;\\&space;1&space;&&space;\sum_{i}c_i&space;=&space;3,&space;\\&space;\end{array}&space;\right." />

  - where *k* = 3 and *d* is a user parameter
- [NK-landscape](http://ncra.ucd.ie/wp-content/uploads/2020/08/SocialLearning_GECCO2019.pdf)
- [W-Model](http://iao.hfuu.edu.cn/images/publications/W2018TWMATBBDOBPIFTBGW.pdf)
