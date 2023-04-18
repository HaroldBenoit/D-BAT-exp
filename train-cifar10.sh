




for alpha in 0.5 
    do 
        for epochs in 200 600
            do
                for ensemble_size in 2 
                    do 
                        python ./src/main.py --ensemble_size $ensemble_size  --config ./cifar.yml --split_semantic_task_idx 0 --split_spurious_task_idx 4 --epochs $epochs --alpha $alpha --group D-BAT-v3 

                    done

            done

    done 