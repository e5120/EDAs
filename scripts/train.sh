. var

# [one_max|four_peaks|deceptive_order_3|deceptive_order_4|knapsack]
target=deceptive_order_3
python ${src_dir}/main.py \
        --objective-type ${target} \
        --setting-path ${root_dir}/settings/${target}.ini \
        --optim-type aff_eda \
        --lam 500 \
        --lr 0.1 \
        --negative-lr 0.075 \
        --replace-rate 0.1 \
        --selection tournament \
        --sampling-rate 0.03 \
        --selection-rate 0.005 \
        --crossover none \
        --theta-cross-prob 0.8 \
        --mutation none \
        --theta-mut-prob 0.02 \
        --theta-mut-shift 0.05 \
        --replacement worst \
        --window-size 10 \
        --train-steps 1000 \
        --trials 3 \
        --seed 1 \
        --logging-step 1 \
        --display-step 1 \
        --save-param-step 100000000
        # --log-dir result/${target}
