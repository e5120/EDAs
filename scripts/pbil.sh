. var

# [one_max|two_min|four_peaks|deceptive_trap|nk_landscape|w_model]
objective=one_max
python ${root_dir}/main.py \
        --objective-type ${objective} \
        --dim 100 \
        --optim-type pbil \
        --lam 32 \
        --lr 0.1 \
        --negative-lr 0.075 \
        --mutation-prob 0.02 \
        --mutation-shift 0.05 \
        --max-num-evals 10000 \
        --trials 3 \
        --seed -1 \
        --display-step 10 \
        # --logging-step 10 \
        # --log-dir log/pbil_${objective}
