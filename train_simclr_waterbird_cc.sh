
## DBAT: D_ood = D_test
python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs 30 --batch_size_train 64 --batch_size_eval 256  --group SIMCLR_CC_WaterBirds_2 --model resnet50SIMCLRv2 --seed 0 --alpha 0.1 --perturb_type ood_is_test  --results_base_folder varying_pretraining 
python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs 30 --batch_size_train 64 --batch_size_eval 256  --group SIMCLR_CC_WaterBirds_2 --model resnet50SIMCLRv2 --seed 1 --alpha 0.1 --perturb_type ood_is_test  --results_base_folder varying_pretraining
python ./src/main.py --config ./configs/waterbirds_cc.yml --epochs 30 --batch_size_train 64 --batch_size_eval 256  --group SIMCLR_CC_WaterBirds_2 --model resnet50SIMCLRv2 --seed 2 --alpha 0.1 --perturb_type ood_is_test  --results_base_folder varying_pretraining

