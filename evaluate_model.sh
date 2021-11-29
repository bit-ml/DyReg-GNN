RAND=$((RANDOM))

MODEL_DIR='./models/'$1
LOG_NAME=$MODEL_DIR'/log_'$RAND

mkdir $MODEL_DIR

RESUME_CKPT='./checkpoints/dyreg_gnn_model_l2l3l4.pth.tar'

args="--name=$1 --visualisation=hsv  --weights=$RESUME_CKPT 
        --npb  --warmup_validate=False 
        --offset_generator=big --dynamic_regions=dyreg  
        --use_rstg=True --bottleneck_graph=True
        --rstg_skip_connection=False --remap_first=True 
        --place_graph=layer2.2_layer3.4_layer4.1 
        --ch_div=4 --graph_residual_type=norm 
        --rnn_type=GRU --aggregation_type=dot --send_layers=1
        --update_layers=0 --combine_by_sum=True --project_at_input=True 
        --tmp_norm_skip_conn=False --init_regions=center 
        --lr 0.001 --batch-size 2 --dataset=somethingv2 --modality=RGB --arch resnet50 
        --test_segments 16 --gd 20 --lr_steps 20 40 --epochs 60 -j 16 
        --dropout 0.000000000000000001 --consensus_type=avg --eval-freq=1   
        --shift --shift_div=8 --shift_place=blockres  --model_dir=$MODEL_DIR  
        --full_size_224=False  --full_res=False --save_kernels=True"


echo $args > $MODEL_DIR'/args_'$RAND

# run on CPU:
# CUDA_VISIBLE_DEVICES="" python -u  test_models.py $args  & tee $LOG_NAME

# run on GPU
CUDA_VISIBLE_DEVICES=1 python -u -m  torch.distributed.launch --nproc_per_node=1  --master_port=6004 test_models.py $args  |& tee $LOG_NAME



