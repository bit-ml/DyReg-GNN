RAND=$((RANDOM))

MODEL_DIR='./models/'$1
LOG_NAME=$MODEL_DIR'/log_'$RAND
mkdir $MODEL_DIR

args="--print-freq=1 --name=$1  
        --npb --offset_generator=big
        --init_skip_zero=False --warmup_validate=False 
        --dynamic_regions=dyreg --init_regions=center
        --place_graph=layer2.2_layer3.4_layer4.1 
        --ch_div=4  --graph_residual_type=norm 
        --rnn_type=GRU --aggregation_type=dot --send_layers=1 
        --use_rstg=True --rstg_skip_connection=False --remap_first=True 
        --update_layers=0 --combine_by_sum=True --project_at_input=True 
        --tmp_norm_skip_conn=False  --bottleneck_graph=True
        --lr 0.001 --batch-size 2 --dataset=somethingv2 --modality=RGB --arch=resnet50 --num_segments 16 --gd 20 
        --lr_type=step --lr_steps 20 30 40  --epochs 60 -j 16 --dropout=0.5 --consensus_type=avg --eval-freq=1  
        --shift_div=8 --shift --shift_place=blockres  --model_dir=$MODEL_DIR"


echo $args > $MODEL_DIR'/args_'$RAND

# run on CPU:
# CUDA_VISIBLE_DEVICES="" python -u  main_standard.py $args  & tee $LOG_NAME

# run on GPU
CUDA_VISIBLE_DEVICES=0 python -u -m  torch.distributed.launch --nproc_per_node=1  --master_port=6004 main_standard.py $args  |& tee $LOG_NAME
