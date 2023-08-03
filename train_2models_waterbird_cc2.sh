#for epochs in 10 30 100
#do
#for alpha in 0.1
#do
#    # python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs $epochs --model resnet50 vit_b_16  --seed 0 --alpha $alpha --perturb_type ood_is_test  --results_base_folder two_models --group two_models_dbat
#    # python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs $epochs --model vit_b_16 resnet50  --seed 0 --alpha $alpha --perturb_type ood_is_test  --results_base_folder two_models --group two_models_dbat
#    python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs $epochs --model resnet50_np resnet50  --seed 0 --alpha $alpha --perturb_type ood_is_test  --results_base_folder two_models --group two_models_dbat
#    python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs $epochs --model resnet50 resnet50_np  --seed 0 --alpha $alpha --perturb_type ood_is_test  --results_base_folder two_models --group two_models_dbat
#    #python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs $epochs --model resnet50 resnet50MocoV2  --seed 0 --alpha $alpha --perturb_type ood_is_test  --results_base_folder two_models --group two_models_dbat
#    python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs $epochs --model resnet50MocoV2 resnet50  --seed 0 --alpha $alpha --perturb_type ood_is_test  --results_base_folder two_models --group two_models_dbat
#
#    #python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs $epochs --model vit_b_16 resnet50_np --seed 0 --alpha $alpha --perturb_type ood_is_test  --results_base_folder two_models --group two_models_dbat
#
#done
#done


#for epochs in 10 30 100
#do
#for alpha in 0.1
#do
#    python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs $epochs --model resnet50_np vit_b_16  --seed 0 --alpha $alpha --perturb_type ood_is_test  --results_base_folder two_models --group two_models_dbat
#    python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs $epochs --model vit_b_16 resnet50_np --seed 0 --alpha $alpha --perturb_type ood_is_test  --results_base_folder two_models --group two_models_dbat
#    python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs $epochs --model resnet50 resnet50MocoV2  --seed 0 --alpha $alpha --perturb_type ood_is_test  --results_base_folder two_models --group two_models_dbat
#
#
#done
#done



#for epochs in 20 30 40 
#do
#for alpha in 1.0 0.1 0.001
#do
#for seed in 0 1 2
#do
#
#    python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs $epochs --model resnet50_np vit_b_16  --seed $seed --alpha $alpha --perturb_type ood_is_test  --results_base_folder two_models --group two_models_dbat_tuning
#
#done
#done
#done


#for first_epoch in 50 75
#do
#for second_epoch in 10 20 30  
#do
#for alpha in 0.1
#do
#for seed in 0 1 2
#do
#
#    python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs $first_epoch $second_epoch  --model resnet50_np resnet50  --seed $seed --alpha $alpha --perturb_type ood_is_test  --results_base_folder two_models --group two_models_dbat_epoch
#
#done
#done
#done
#done


#for epoch in 10 30 60
#do
#for alpha in 0.1
#do
#for seed in 0 1 2
#do
#
#    python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs $epoch --ensemble_size 3 --model resnet50_np resnet50 resnet50  --seed $seed --alpha $alpha --perturb_type ood_is_test  --results_base_folder three_models --group two_models_dbat_epoch
#
#done
#done
#done


for epochs in 30 60 100
do
for alpha in 0.1
do
for seed in 0 1 2
do

    python ./src/main.py --config ./configs/waterbirds_cc.yml  --inverse_correlation  --epochs $epochs --model resnet50_np vit_b_16  --seed $seed --alpha $alpha --perturb_type ood_is_test  --results_base_folder two_models_IC --group two_models_dbat_IC

done
done
done

