from opts import parser
global args
args = parser.parse_args()

def return_config_resnet13():
    graph_params = {}
    graph_params[1] = {'in_channels' : 64,  'H' : 32, 'node_dim' : 2 * 64 // args.ch_div, 'project_i3d' : True, 'name': 'layer1'}
    graph_params[2] = {'in_channels' : 128, 'H' : 16, 'node_dim' : 128 // args.ch_div, 'project_i3d' : True, 'name': 'layer2'}
    graph_params[3] = {'in_channels' : 256, 'H' : 8, 'node_dim' : 256 // args.ch_div, 'project_i3d' : True, 'name': 'layer3'}
    out_pool_size = 8
    # 256
    return graph_params, out_pool_size

def return_config_resnet18():
    graph_params = {}
    if args.bottleneck_graph:
        if args.full_res:
            graph_params[1] = {'in_channels' : 64 // args.ch_div,  'iH' : 56,  'H' : 64, 'node_dim' : 64 // args.ch_div, 'name': 'layer1'}
            graph_params[2] = {'in_channels' : 128// args.ch_div,  'iH' : 28,  'H' : 32, 'node_dim' : 128 // args.ch_div, 'name': 'layer2'}
            graph_params[3] = {'in_channels' : 256// args.ch_div,  'iH' : 14, 'H' : 16, 'node_dim' : 256 // args.ch_div, 'name': 'layer3'}
            graph_params[4] = {'in_channels' : 512// args.ch_div,  'iH' : 7, 'H' : 8,  'node_dim' : min(256,512 // args.ch_div), 'project_i3d' : True, 'name': 'layer4'}
            out_pool_size = 8
        else:
            graph_params[1] = {'in_channels' : 64 // args.ch_div, 'H' : 56, 'iH' : 56, 'node_dim' : 64 // args.ch_div, 'name': 'layer1'}
            graph_params[2] = {'in_channels' : 128// args.ch_div, 'H' : 28, 'iH' : 28, 'node_dim' : 128 // args.ch_div, 'name': 'layer2'}
            graph_params[3] = {'in_channels' : 256// args.ch_div, 'H' : 14, 'iH' : 14, 'node_dim' : 256 // args.ch_div, 'name': 'layer3'}
            graph_params[4] = {'in_channels' : 512// args.ch_div, 'H' : 7, 'iH' : 7,  'node_dim' : min(256,512 // args.ch_div), 'project_i3d' : True, 'name': 'layer4'}
            out_pool_size = 7
    
        for i in [1,2,3]:
            graph_params[i]['project_i3d'] = (args.offset_generator == 'glore') #True for glore, False for fishnet
    else:
        graph_params[1] = {'in_channels' : 64,  'H' : 56, 'node_dim' : 64 // args.ch_div, 'project_i3d' : True, 'name': 'layer1'}
        graph_params[2] = {'in_channels' : 128,  'H' : 28, 'node_dim' : 128 // args.ch_div, 'project_i3d' : True, 'name': 'layer2'}
        graph_params[3] = {'in_channels' : 256, 'H' : 14, 'node_dim' : 256 // args.ch_div, 'project_i3d' : True, 'name': 'layer3'}
        graph_params[4] = {'in_channels' : 512, 'H' : 7,  'node_dim' : min(512,512 // args.ch_div), 'project_i3d' : True, 'name': 'layer4'}
        out_pool_size = 7
    
    return graph_params, out_pool_size

def return_config_resnet34():
    graph_params = {}
   
    graph_params[1] = {'in_channels' : 64,  'H' : 56, 'node_dim' : 64 // args.ch_div, 'project_i3d' : True, 'name': 'layer1'}
    graph_params[2] = {'in_channels' : 128,  'H' : 28, 'node_dim' : 128 // args.ch_div, 'project_i3d' : True, 'name': 'layer2'}
    graph_params[3] = {'in_channels' : 256, 'H' : 14, 'node_dim' : 256 // args.ch_div, 'project_i3d' : True, 'name': 'layer3'}
    graph_params[4] = {'in_channels' : 512, 'H' : 7,  'node_dim' : min(512,512 // args.ch_div), 'project_i3d' : True, 'name': 'layer4'}
    out_pool_size = 7
    
    return graph_params, out_pool_size



def return_config_wide_resnet50_2():
    graph_params = {}
   
    graph_params[1] = {'in_channels' : 512 // args.ch_div, 'H' : 56, 'iH' : 56, 'node_dim' : 512 // args.ch_div, 'name': 'layer1'}
    graph_params[2] = {'in_channels' : 1024// args.ch_div, 'H' : 28, 'iH' : 28, 'node_dim' : 1024 // args.ch_div, 'name': 'layer2'}
    graph_params[3] = {'in_channels' : 2048// args.ch_div, 'H' : 14, 'iH' : 14, 'node_dim' : 2048 // args.ch_div, 'name': 'layer3'}
    graph_params[4] = {'in_channels' : 4096// args.ch_div, 'H' : 7, 'iH' : 7,  'node_dim' : min(512,4096 // args.ch_div), 'project_i3d' : True, 'name': 'layer4'}
    
    
    out_pool_size = 7

    for i in [1,2,3]:
        graph_params[i]['project_i3d'] = (args.offset_generator == 'glore') #True for glore, False for fishnet
    
    
    return graph_params, out_pool_size
def return_config_resnet50():
    graph_params = {}
    out_pool_size = 0
    if args.bottleneck_graph:
        
        if args.full_res:
            graph_params[1] = {'in_channels' : 256 // args.ch_div,  'iH' : 56,  'H' : 64, 'node_dim' : 256 // args.ch_div, 'name': 'layer1'}
            graph_params[2] = {'in_channels' : 512// args.ch_div,  'iH' : 28,  'H' : 32, 'node_dim' : 512 // args.ch_div, 'name': 'layer2'}
            graph_params[3] = {'in_channels' : 1024// args.ch_div,  'iH' : 14, 'H' : 16, 'node_dim' : 1024 // args.ch_div, 'name': 'layer3'}
            graph_params[4] = {'in_channels' : 2048// args.ch_div,  'iH' : 7, 'H' : 8,  'node_dim' : min(256,2048 // args.ch_div), 'project_i3d' : True, 'name': 'layer4'}
            out_pool_size = 8

        else:
            graph_params[1] = {'in_channels' : 256 // args.ch_div, 'H' : 56, 'iH' : 56, 'node_dim' : 256 // args.ch_div, 'name': 'layer1'}
            graph_params[2] = {'in_channels' : 512// args.ch_div, 'H' : 28, 'iH' : 28, 'node_dim' : 512 // args.ch_div, 'name': 'layer2'}
            graph_params[3] = {'in_channels' : 1024// args.ch_div, 'H' : 14, 'iH' : 14, 'node_dim' : 1024 // args.ch_div, 'name': 'layer3'}
            graph_params[4] = {'in_channels' : 2048// args.ch_div, 'H' : 7, 'iH' : 7,  'node_dim' : min(256,2048 // args.ch_div), 'project_i3d' : True, 'name': 'layer4'}
            out_pool_size = 7
    
        for i in [1,2,3]:
            graph_params[i]['project_i3d'] = (args.offset_generator == 'glore') #True for glore, False for fishnet
    
    
    else:
        # TO BE REEMOVED
        graph_params[1] = {'in_channels' : 256,  'H' : 56, 'node_dim' : 256 // args.ch_div, 'project_i3d' : True, 'name': 'layer1'}
        graph_params[2] = {'in_channels' : 512,  'H' : 28, 'node_dim' : 512 // args.ch_div, 'project_i3d' : True, 'name': 'layer2'}
        graph_params[3] = {'in_channels' : 1024, 'H' : 14, 'node_dim' : 1024 // args.ch_div, 'project_i3d' : True, 'name': 'layer3'}
        graph_params[4] = {'in_channels' : 2048, 'H' : 7,  'node_dim' : min(512,2048 // args.ch_div), 'project_i3d' : True, 'name': 'layer4'}

    return graph_params, out_pool_size

def return_config_resnet101():
    graph_params = {}
    if args.bottleneck_graph:

            graph_params[1] = {'in_channels' : 256 // args.ch_div, 'H' : 56, 'iH' : 56, 'node_dim' : 256 // args.ch_div, 'name': 'layer1'}
            graph_params[2] = {'in_channels' : 512// args.ch_div, 'H' : 28, 'iH' : 28, 'node_dim' : 512 // args.ch_div, 'name': 'layer2'}
            graph_params[3] = {'in_channels' : 1024// args.ch_div, 'H' : 14, 'iH' : 14, 'node_dim' : 1024 // args.ch_div, 'name': 'layer3'}
            graph_params[4] = {'in_channels' : 2048// args.ch_div, 'H' : 7, 'iH' : 7,  'node_dim' : min(256,2048 // args.ch_div), 'project_i3d' : True, 'name': 'layer4'}
            out_pool_size = 7

            for i in [1,2,3]:
                graph_params[i]['project_i3d'] = (args.offset_generator == 'glore') #True for glore, False for fishnet
    else:
        graph_params[1] = {'in_channels' : 256,  'H' : 56, 'node_dim' : 256 // args.ch_div, 'project_i3d' : True, 'name': 'layer1'}
        graph_params[2] = {'in_channels' : 512,  'H' : 28, 'node_dim' : 512 // args.ch_div, 'project_i3d' : True, 'name': 'layer2'}
        graph_params[3] = {'in_channels' : 1024, 'H' : 14, 'node_dim' : 1024 // args.ch_div, 'project_i3d' : True, 'name': 'layer3'}
        graph_params[4] = {'in_channels' : 2048, 'H' : 7,  'node_dim' : min(512,2048 // args.ch_div), 'project_i3d' : True, 'name': 'layer4'}
    out_pool_size = 7
    if args.full_res:
        out_pool_size = 8
    return graph_params, out_pool_size





def get_models_config():
    graph_params = {}
    if args.arch == 'resnet13':
        graph_params, out_pool_size = return_config_resnet13()
    if args.arch == 'resnet18':
        graph_params, out_pool_size = return_config_resnet18()
    elif args.arch == 'resnet34':
        graph_params, out_pool_size = return_config_resnet34() 
    elif args.arch == 'resnet50':
        graph_params, out_pool_size = return_config_resnet50()
    elif args.arch == 'wide_resnet50_2':
        graph_params, out_pool_size = return_config_wide_resnet50_2()
    elif args.arch == 'resnet101':
        graph_params, out_pool_size = return_config_resnet101()
    out_num_ch = 2048
    return graph_params, out_pool_size, out_num_ch, None