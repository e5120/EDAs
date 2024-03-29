# [one_max|two_min|four_peaks|deceptive_trap|nk_landscape|w_model]
objective=one_max
python ../main.py \
        --objective-type ${objective} \
        --dim 100 \
        --optim-type umda \
        --lam 64 \
        --lr 0.5 \
        --selection tournament \
        --selection-rate 0.5 \
        --tournament-size 2 \
        --max-num-evals 10000 \
        --trials 3 \
        --seed -1 \
        --display-step 10 \
        # --logging-step 1 \
        # --log-dir log/cga_${objective}
