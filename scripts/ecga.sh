. var

# [one_max|two_min|four_peaks|deceptive_trap|nk_landscape|w_model]
objective=deceptive_trap
python ${root_dir}/main.py \
        --objective-type ${objective} \
        --dim 30 \
        -k 3 \
        --optim-type ecga \
        --lam 500 \
        --selection tournament \
        --selection-rate 0.5 \
        --tournament-size 2 \
        --replacement restricted \
        --replace-rate 0.5 \
        --window-size 2 \
        --max-num-evals 10000 \
        --trials 3 \
        --seed -1 \
        --logging-step 1 \
        --display-step 1 \
        # --log-dir log/ecga_${objective}
