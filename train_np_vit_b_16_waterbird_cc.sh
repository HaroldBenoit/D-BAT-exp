for epochs in 100
do
for alpha in 0.1
do
    for seed in 0 1 2 

    do 
    python ./src/main.py --config ./configs/waterbirds_cc_np.yml --epochs $epochs --batch_size_train 64 --batch_size_eval 64 --no_wandb --model vit_b_16 --seed $seed --alpha $alpha --perturb_type ood_is_test  --results_base_folder varying_pretraining 
    done
done
done

