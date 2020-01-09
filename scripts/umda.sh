. var

# [one_max|four_peaks|deceptive_order_3|deceptive_order_4|knapsack]
target=one_max
python ${src_dir}/main.py \
        --objective-type ${target} \
        --setting-path ${root_dir}/settings/${target}.ini \
        --optim-type umda \
        --lam 100 \
        --lr 0.1 \
        --selection tournament \
        --sampling-rate 0.3 \
        --selection-rate 0.005 \
        --train-steps 1000 \
        --trials 3 \
        --seed 1 \
        --logging-step 1 \
        --display-step 1 \
        --save-param-step 100000000
        # --log-dir result/${target}
