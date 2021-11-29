
import itertools

var_params = {}

# var_params['aux_loss'] = ['inter_videos_all', 'inter_videos_content', 'inter_nodes_all', 'inter_nodes_content']
# var_params['aux_loss'] = ['global_foreground_pos_background_neg', 
#                                 'global_foreground_pos_foreground_neg', 
#                                 'global_foreground_pos_foreground_background_neg', 
#                                 'global_foreground_pos_foreground_background_neg_rand']


var_params['contrastive_mlp'] = [False]

var_params['aux_loss'] = ['nodes_temporal_matching']
var_params['contrastive_alpha'] = [0.1, 1.0]
var_params['contrastive_temp'] = [0.1, 0.5, 1.0, 2.0]
# var_params['contrastive_temp'] = [0.05]

# var_params['place_graph'] = ['layer2.1_layer3.1_layer4.1']
# var_params['place_graph'] = ['layer3.1']

somelists = [val for k,val in var_params.items()]


for element in itertools.product(*somelists):
    print('PARAMS')
    args = ''
    keys = list(var_params.keys())

    name_model = 'model_contrastive'
    for i,k in enumerate(keys):
        name_model += f'_{k}_{element[i]}'
        args += f' --{k}={element[i]}'
    args += f' --name={name_model}'
    # print(args)

    args = args + f" --place_graph=layer2.1_layer3.1_layer4.1 --contrastive_type=nodes_temporal --rstg_combine=plus --node_confidence=none  --use_detector=False --compute_iou=False --tmp_fix_kernel_grads=False --create_order=False --distill=False --distributed=True --bottleneck_graph=False --scaled_distill=False --alpha_distill=1.0 --glore_pos_emb_type=sinusoidal --npb --offset_generator=fishnet --predefined_augm=False --freeze_backbone=False --init_skip_zero=False --warmup_validate=False --smt_split=classic --use_rstg=True --tmp_init_different=0.0 --rstg_skip_connection=False --remap_first=True --isolate_graphs=False --dynamic_regions=constrained_fix_size --ch_div=4  --graph_residual_type=norm --rnn_type=GRU --aggregation_type=dot --send_layers=1 --update_layers=0 --combine_by_sum=True --project_at_input=True --tmp_norm_after_project_at_input=True --tmp_norm_skip_conn=False --tmp_norm_before_dynamic_graph=False --init_regions=center --tmp_increment_reg=True --lr 0.01 --batch-size 14 --dataset=cater --modality=RGB --arch=resnet18 --num_segments 13 --gd 20 --lr_type=step --lr_steps 75 125 200  --epochs 150 -j 16 --dropout=0.5 --consensus_type=avg --eval-freq=1  --shift_div=8 --shift --shift_place=blockres  --model_dir=$MODEL_DIR"

    
    script_name = f'auto_scripts/start_' + name_model + '.sh'

    with open(script_name, 'w') as f:
        f.write('RAND=$((RANDOM))\n')
        f.write(f'MODEL_DIR=\'/models/graph_models/models_cater_dynamic_pytorch/exp_contrastive_global/{name_model}\'\n')

        f.write('LOG_NAME=$MODEL_DIR\'/log_\'$RAND\n')
        f.write('mkdir $MODEL_DIR\n')
        f.write('\n')
        f.write(f'args="{args}"\n')
        f.write('\n')


        f.write('cp ./train.py $MODEL_DIR\'/train.py_\'$RAND\n')
        f.write('mkdir $MODEL_DIR\'/code/\'\n')
        f.write('mkdir $MODEL_DIR\'/code/ops_\'$RAND\'/\'\n')
        f.write('cp -r ./ops/  $MODEL_DIR\'/code/ops_\'$RAND\'/\'\n')
        f.write('cp -r ./main.py  $MODEL_DIR\'/code/\'\n')
        f.write('echo $args > $MODEL_DIR\'/args_\'$RAND\n')

        f.write('\n')
        f.write('\n')

        f.write(f'CUDA_VISIBLE_DEVICES=$1 python -u -m  torch.distributed.launch --nproc_per_node=1  --master_port=$2 main_contrastive.py $args  |& tee $LOG_NAME\n')
