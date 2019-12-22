. var

# [one_max|four_peaks|deceptive_order_3|deceptive_order_4|knapsack]
target=one_max
python ${src_dir}/main.py \
        --objective-type ${target} \
        --setting-path ${root_dir}/settings/${target}.ini \
        --optim-type ga \
        --lam 100 \
        --selection none \
        --crossover two_point \
        --crossover-prob 0.7 \
        --mutation mutation \
        --mutation-prob 0.001 \
        --replacement trunc \
        --train-steps 2000 \
        --trials 10 \
        --logging-step 1 \
        --display-step 1 \
        --save-param-step 100000000
        # --log-dir result/${target}
