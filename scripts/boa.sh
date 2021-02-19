. var

# [one_max|two_min|four_peaks|deceptive_trap|nk_landscape|w_model]
objective=deceptive_trap
python ${root_dir}/main.py \
        --objective-type ${objective} \
        --dim 30 \
        -k 3 \
        --optim-type boa \
        --lam 800 \
        --selection top \
        --selection-rate 0.5 \
        --replacement truncation \
        --replace-rate 0.5 \
        --metric bic \
        --max-num-evals 10000 \
        --trials 3 \
        --seed -1 \
        --display-step 10 \
        # --logging-step 10 \
        # --log-dir log/boa_${objective}
