# EDAs
Implementation of Estimation of Distribution Algorithm (EDA)

## 推奨値
|                                                       手法                                                       |                                          推奨パラメータ                                           |
|:----------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|
|                              [PBIL](https://apps.dtic.mil/docs/citations/ADA282654)                              |               --lam 32 --lr 0.1 --negative-lr 0.75 --mut-prob 0.02 --mut-shif 0.05                |
|                                 [UMDA](http://www.muehlenbein.org/estbin96.pdf)                                  |                              --lam 32 --lr 0.1 --selection roulette                               |
|                                [CGA](https://ieeexplore.ieee.org/document/797971)                                |                                             --lam 64                                              |
|     [MIMIC](https://papers.nips.cc/paper/1328-mimic-finding-optima-by-estimating-probability-densities.pdf)      |                          --lam 128 --replace-rate 0.1 --replacement top                           |
| [ECGA](https://www.researchgate.net/publication/2460502_Linkage_Learning_via_Probabilistic_Modeling_in_the_ECGA) | --lam 500 --replacement restricted --selection tournament   --window-size 10 --sampling-rate 0.03 |
|                              [AffEDA](https://ieeexplore.ieee.org/document/6793952)                              | --lam 500 --replacement restricted --selection tournament   --window-size 10 --sampling-rate 0.03 |
