. var

# [one_max|four_peaks|deceptive_order_3|deceptive_order_4|knapsack]
target=deceptive_order_3
python ${src_dir}/main.py \
        --objective-type ${target} \
        --setting-path ${root_dir}/settings/${target}.ini \
        --optim-type ecga \
        --lam 500 \
        --train-steps 2000 \
        --trials 10 \
        --seed 1 \
        --logging-step 1 \
        --display-step 1 \
        --save-param-step 100000000
        # --log-dir result/${target}
