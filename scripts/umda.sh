. var

# [one_max|two_min|four_peaks|deceptive_trap|nk_landscape|w_model]
objective=one_max
python ${root_dir}/main.py \
        --objective-type ${objective} \
        --dim 100 \
        --optim-type umda \
        --lam 64 \
        --lr 0.5 \
        --selection tournament \
        --tournament-size 2 \
        --sampling-rate 0.5 \
        --max-num-evals 10000 \
        --trials 3 \
        --seed -1 \
        --logging-step 1 \
        --display-step 1 \
        # --log-dir log/cga_${objective}
