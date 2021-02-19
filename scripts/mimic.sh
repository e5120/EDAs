. var

# [one_max|two_min|four_peaks|deceptive_trap|nk_landscape|w_model]
objective=one_max
python ${root_dir}/main.py \
        --objective-type ${objective} \
        --dim 50 \
        --optim-type mimic \
        --lam 128 \
        --replacement restricted \
        --replace-rate 0.5 \
        --window-size 2 \
        --max-num-evals 10000 \
        --trials 3 \
        --seed -1 \
        --display-step 10 \
        # --logging-step 10 \
        # --log-dir log/ecga_${objective}
