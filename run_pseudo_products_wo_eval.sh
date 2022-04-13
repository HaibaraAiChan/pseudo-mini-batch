#!/bin/bash

File=pseudo_mini_batch_range_products_sage.py

model=sage
seed=1236 
setseed=True
GPUmem=True
lr=0.01
dropout=0.5
# layers=3
Data=ogbn-products
# hidden=64
run=1
batch_size=(98308 49154 24577 12289 6145 3073 1537 769)
# Aggre=lstm
# nb=1
# pMethod=range
savePath=../logs/sage/1_runs/pure_train/ogbn_products
savePath=/home/cc/graph_partition_multi_layers/pseudo_mini_batch_full_batch/logs/sage/1_runs/pure_train/ogbn_prodcuts
hiddenList=(32 64 128 256)


pMethodList=(random range) 
AggreList=(mean lstm)
epoch=2
logIndent=1

fan_out_list=(25,35,40 25,35,80 25,70,80 50,70,80)
layersList=(3)

for Aggre in ${AggreList[@]}
do      
        mkdir ${savePath}/${Aggre}/
        # echo ${savePath}/${Aggre}/
        for pMethod in ${pMethodList[@]}
        do      
                mkdir ${savePath}/${Aggre}/${pMethod}
                for layers in ${layersList[@]}
                do      
                        mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/
                        for hidden in ${hiddenList[@]}
                        do
                                mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/
                                for fan_out in ${fan_out_list[@]}
                                do
                                        nb=1
                                        for bs in ${batch_size[@]}
                                        do
                                                nb=$(($nb*2))
                                                # nb=32
                                                mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
                                                logPath=${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
                                                # mkdir $logPath
                                                echo $logPath
                                                echo 'number of batches'
                                                echo $nb
                                                python $File \
                                                --dataset $Data \
                                                --aggre $Aggre \
                                                --seed $seed \
                                                --setseed $setseed \
                                                --GPUmem $GPUmem \
                                                --selection-method $pMethod \
                                                --batch-size $bs \
                                                --lr $lr \
                                                --num-runs $run \
                                                --num-epochs $epoch \
                                                --num-layers $layers \
                                                --num-hidden $hidden \
                                                --dropout $dropout \
                                                --fan-out $fan_out \
                                                --log-indent $logIndent \
                                                &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_nb_${nb}_r_${run}_ep_${epoch}.log
                                        done
                                done
                        done
                done
        done
done

# fan_out_list=(25,35,40,40)
# layersList=(4)
# for Aggre in ${AggreList[@]}
# do      
#         mkdir ${savePath}/${Aggre}/
#         # echo ${savePath}/${Aggre}/
#         for pMethod in ${pMethodList[@]}
#         do      
#                 mkdir ${savePath}/${Aggre}/${pMethod}
#                 for layers in ${layersList[@]}
#                 do      
#                         mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/
#                         for hidden in ${hiddenList[@]}
#                         do
#                                 mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/
#                                 for fan_out in ${fan_out_list[@]}
#                                 do
#                                         nb=1
#                                         for bs in ${batch_size[@]}
#                                         do
#                                                 nb=$(($nb*2))
#                                                 # nb=32
#                                                 mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
#                                                 logPath=${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
#                                                 # mkdir $logPath
#                                                 echo $logPath
#                                                 echo 'number of batches'
#                                                 echo $nb
#                                                 python $File \
#                                                 --dataset $Data \
#                                                 --aggre $Aggre \
#                                                 --seed $seed \
#                                                 --setseed $setseed \
#                                                 --GPUmem $GPUmem \
#                                                 --selection-method $pMethod \
#                                                 --batch-size $bs \
#                                                 --lr $lr \
#                                                 --num-runs $run \
#                                                 --num-epochs $epoch \
#                                                 --num-layers $layers \
#                                                 --num-hidden $hidden \
#                                                 --dropout $dropout \
#                                                 --fan-out $fan_out \
#                                                 --log-indent $logIndent \
#                                                 &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_nb_${nb}_r_${run}_ep_${epoch}.log
#                                         done
#                                 done
#                         done
#                 done
#         done
# done

# layersList=(5)
# fan_out_list=(25,35,40,40,40)
# for Aggre in ${AggreList[@]}
# do      
#         mkdir ${savePath}/${Aggre}/
#         # echo ${savePath}/${Aggre}/
#         for pMethod in ${pMethodList[@]}
#         do      
#                 mkdir ${savePath}/${Aggre}/${pMethod}
#                 for layers in ${layersList[@]}
#                 do      
#                         mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/
#                         for hidden in ${hiddenList[@]}
#                         do
#                                 mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/
#                                 for fan_out in ${fan_out_list[@]}
#                                 do
#                                         nb=1
#                                         for bs in ${batch_size[@]}
#                                         do
#                                                 nb=$(($nb*2))
#                                                 # nb=32
#                                                 mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
#                                                 logPath=${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
#                                                 # mkdir $logPath
#                                                 echo $logPath
#                                                 echo 'number of batches'
#                                                 echo $nb
#                                                 python $File \
#                                                 --dataset $Data \
#                                                 --aggre $Aggre \
#                                                 --seed $seed \
#                                                 --setseed $setseed \
#                                                 --GPUmem $GPUmem \
#                                                 --selection-method $pMethod \
#                                                 --batch-size $bs \
#                                                 --lr $lr \
#                                                 --num-runs $run \
#                                                 --num-epochs $epoch \
#                                                 --num-layers $layers \
#                                                 --num-hidden $hidden \
#                                                 --dropout $dropout \
#                                                 --fan-out $fan_out \
#                                                 --log-indent $logIndent \
#                                                 &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_nb_${nb}_r_${run}_ep_${epoch}.log
#                                         done
#                                 done
#                         done
#                 done
#         done
# done

# layersList=(6)
# fan_out_list=(25,35,40,40,40,40)
# for Aggre in ${AggreList[@]}
# do      
#         mkdir ${savePath}/${Aggre}/
#         # echo ${savePath}/${Aggre}/
#         for pMethod in ${pMethodList[@]}
#         do      
#                 mkdir ${savePath}/${Aggre}/${pMethod}
#                 for layers in ${layersList[@]}
#                 do      
#                         mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/
#                         for hidden in ${hiddenList[@]}
#                         do
#                                 mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/
#                                 for fan_out in ${fan_out_list[@]}
#                                 do
#                                         nb=1
#                                         for bs in ${batch_size[@]}
#                                         do
#                                                 nb=$(($nb*2))
#                                                 # nb=32
#                                                 mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
#                                                 logPath=${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
#                                                 # mkdir $logPath
#                                                 echo $logPath
#                                                 echo 'number of batches'
#                                                 echo $nb
#                                                 python $File \
#                                                 --dataset $Data \
#                                                 --aggre $Aggre \
#                                                 --seed $seed \
#                                                 --setseed $setseed \
#                                                 --GPUmem $GPUmem \
#                                                 --selection-method $pMethod \
#                                                 --batch-size $bs \
#                                                 --lr $lr \
#                                                 --num-runs $run \
#                                                 --num-epochs $epoch \
#                                                 --num-layers $layers \
#                                                 --num-hidden $hidden \
#                                                 --dropout $dropout \
#                                                 --fan-out $fan_out \
#                                                 --log-indent $logIndent \
#                                                 &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_nb_${nb}_r_${run}_ep_${epoch}.log
#                                         done
#                                 done
#                         done
#                 done
#         done
# done