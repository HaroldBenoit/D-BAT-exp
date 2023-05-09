
## DBAT: D_ood = D_test
python ./src/main.py --config ./configs/waterbirds_cc.yml --batch_size_train 64 --batch_size_eval 256 --group Robust_CC_WaterBirds --model robust_resnet50 --seed 0 --alpha 0.0001 --perturb_type ood_is_test  --results_base_folder varying_pretraining 
python ./src/main.py --config ./configs/waterbirds_cc.yml --batch_size_train 64 --batch_size_eval 256 --group Robust_CC_WaterBirds --model robust_resnet50 --seed 1 --alpha 0.0001 --perturb_type ood_is_test  --results_base_folder varying_pretraining
python ./src/main.py --config ./configs/waterbirds_cc.yml --batch_size_train 64 --batch_size_eval 256 --group Robust_CC_WaterBirds --model robust_resnet50 --seed 2 --alpha 0.0001 --perturb_type ood_is_test  --results_base_folder varying_pretraining

