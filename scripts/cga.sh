# [one_max|two_min|four_peaks|deceptive_trap|nk_landscape|w_model]
objective=one_max
python ../main.py \
        --objective-type ${objective} \
        --dim 50 \
        --optim-type cga \
        --lam 100 \
        --max-num-evals 10000 \
        --trials 3 \
        --seed -1 \
        --display-step 10 \
        # --logging-step 10 \
        # --log-dir log/cga_${objective}
