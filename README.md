# EDAs
Implementation of Estimation of Distribution Algorithms (EDAs).

EDAs is a framework to optimize black-box discrete optimization problems.  
The algorithm of EDAs is as follows.
1. Initialize a population $P$ whose size is $\lambda$.
2. Construct a population $S$ which includes promising solutions in $P$.
3. Build a explicit probabilistic model $M$ based on $S$.
4. Generate new $\lambda_{\textrm{candidate}}$ solutions from $M$ to construct a population $O$.
5. The solutions in $P$ is replaced with those of $O$.
6. If termination conditions are met, then the algorithm is terminated, else go to (2).

## How to install
1. `git clone https://github.com/e5120/EDAs.git`
2. `cd EDAs`
3. `pip install -r requirements.txt`
4. `pip install -e .`

## How to use
1. `cd scripts`
2. Rewrite a script file `xxx.sh`, if necessary.
3. Execute a command `bash xxx.sh`.
4. Rewrite a script file `plot.sh`, if necessary.
5. Execute a command `bash plot.sh` to visualize the results of experiments.

## Methods
- [PBIL](https://apps.dtic.mil/docs/citations/ADA282654)
- [UMDA](http://www.muehlenbein.org/estbin96.pdf)
- [CGA](https://ieeexplore.ieee.org/document/797971)
- [MIMIC](https://papers.nips.cc/paper/1328-mimic-finding-optima-by-estimating-probability-densities.pdf)
- [ECGA](https://www.researchgate.net/publication/2460502_Linkage_Learning_via_Probabilistic_Modeling_in_the_ECGA)
- [AffEDA](https://ieeexplore.ieee.org/document/6793952)
- [BOA](https://dl.acm.org/doi/pdf/10.5555/2933923.2933973)

## Objective function
The search space is $D$-dimensional bit-strings $\bm{c}\in \{0,1\}^D$.
- OneMax: $f(\bm{c})=\sum_{i=1}^Dc_i$
- TwoMin: $f(\bm{c}, \bm{y})=\min(\sum_{i=1}^D|c_i-y_i|,\sum_{i=1}^D|(1-c_i)-y_i|)$
  - $\bm{y}$ is $D$-dimensional bit-strings which is generated randomly in advance.
- Four-peaks: $f(\bm{c})=\max(o(\bm{c}),z(\bm{c}))+$REWARD
  - $o(\bm{c})$ is the number of contiguous ones starting in Position 1.
  - $z(\bm{c})$ is the number of contiguous zeros ending in Position $D$.
  - if $o(\bm{c}) > T$ and $z(\bm{c}) > T$, then REWARD is $D$, else REWARD is 0.
    - $T$ is a user parameter.
- Deceptive-k Trap
- NK-landscape
- W-Model
