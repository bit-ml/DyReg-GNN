import numpy as np
import pdb
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from PIL import Image
import PIL
from moviepy.editor import ImageSequenceClip
import matplotlib
matplotlib.use("Agg")
global args
from opts import parse_args
args = parse_args()


def offsets_to_boxes(offsets, frame_size=224):
    pred_boxes_dict = {}
    places = args.place_graph.replace('layer','').split('_')
    all_positions = []
    for place in places:
        layer = int(place.split('.')[0])
        block = int(place.split('.')[1])
        pred_position = f'layer{layer}_{block}'
        lb = (layer,block)
        all_positions.append(lb)
    for (layer,block) in all_positions:
        pred_position = f'layer{layer}_block{block}'
        num_layer = int(layer)
        pred_offset = offsets[pred_position].squeeze().detach().cpu().numpy()
        pred_offsets = pred_offset / args.graph_params[num_layer]['H'] * frame_size #224.0
        regions_h = pred_offsets[:,:,0]
        regions_w = pred_offsets[:,:,1]
        kernel_center_h = pred_offsets[:,:,2]
        kernel_center_w = pred_offsets[:,:,3]

        y1 = np.minimum(frame_size-1,np.maximum(0,kernel_center_h - regions_h))
        x1 = np.minimum(frame_size-1,np.maximum(0,kernel_center_w - regions_w))
        y2 = np.minimum(frame_size-1, kernel_center_h + regions_h)
        x2 = np.minimum(frame_size-1, kernel_center_w + regions_w)

        pred_boxes = np.stack([y1, x1, y2, x2], axis=-1)
        pred_boxes_dict[f'layer{layer}_{block}'] = pred_boxes
    return pred_boxes_dict



def draw_box(frame, box, color=[1,1,1]):
    H = args.input_size
    (h,w,dh,dw) = box

    left = max(0,int(dw-w))
    right = min(int(dw+w),H-1)
    up = max(int(dh-h),0)
    down = min(int(dh+h), H-1)

    # print(left,right,up,down)
    frame[left,up:down] = color#(0,0,1)

    frame[right,up:down] = color#(0,0,1)
    frame[left:right, up] = color#(0,0,1)
    frame[left:right, down] = color#(0,0,1)
    return frame

def draw_box_gt(frame, target_boxes, colors=[1,1,1], partial=False, line_width=1):
    if len(colors) < target_boxes.shape[0]:
        colors = colors * target_boxes.shape[0]
    for b in range(target_boxes.shape[0]):
        if partial and b % 3 != 0:
            continue
        box = target_boxes[b]
        
        top_h = int(box[0])
        left_w = int(box[1])
        bot_h = int(box[2])
        right_w = int(box[3])
        frame[top_h:bot_h, left_w:left_w + line_width] = colors[b]
        frame[top_h:bot_h, right_w:right_w + line_width] = colors[b]

        frame[top_h: top_h + line_width, left_w : right_w] = colors[b]
        frame[bot_h : bot_h + line_width, left_w : right_w] = colors[b]

    return frame  
def save_grad_cam(input, grad_cams, folder, name='initial'):

    import cv2
    folder_dir = folder + f'/viz_grad_cam_{name}/'
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)

    video_input_val = input.detach().cpu().numpy()
    video_input_val = np.reshape(video_input_val, [args.batch_size, -1, 3 , video_input_val.shape[3],video_input_val.shape[4]])
    video_input_val = np.transpose(video_input_val, [0,1,3,4,2])
    for ind_grad, gr in enumerate(grad_cams):
        name = gr[0]
        grad_cam = gr[1]
        grad_cam = grad_cam.view((-1,16) + grad_cam.shape[1:])
        num_saved_videos = 2
        for video_idx in range(min(num_saved_videos, args.batch_size)):
            max_t = 16
            all_frames = []
            for tt in range(max_t):
                if 'resnet50_smt_else' in args.arch:
                    real_tt = 2*tt
                else:
                    real_tt = tt
        
                frame = video_input_val[video_idx,real_tt,:,:]
                if args.modality == 'RGB':
                    frame = frame * 57.375 + 114.75
     
                grad_c = grad_cam[video_idx, tt].cpu().detach().numpy()

                grad_min = grad_c.min()
                grad_max = (grad_c - grad_min).max()
                grad_c =  (grad_c - grad_min) / grad_max
                grad_c = (grad_c * 255).astype(np.uint8)
                grad_c = np.array(Image.fromarray(grad_c).resize((frame.shape[0],frame.shape[1]), resample=PIL.Image.BILINEAR))

                heatmap = cv2.applyColorMap(grad_c, cv2.COLORMAP_JET)
                cam = np.float32(heatmap) + np.float32(frame)
                cam = cam / cam.max()

                combined_img =np.concatenate((np.uint8(frame), np.uint8(255 * cam)), axis=1)
                cv2.imwrite(os.path.join(folder_dir, f'video_{video_idx}_frame_{tt}_{ind_grad}_grad_{name}.jpg'), combined_img)


def save_mean_kernels(all_kernels,epoch=0,folder=''):
    places = args.place_graph.replace('layer','').split('_')
    placement_all_models = []
    for place in places:
        layer = int(place.split('.')[0])
        block = int(place.split('.')[1])
        placement_all_models.append(f'layer{layer}_block{block}')

    for placement in placement_all_models:
        kernel_val = all_kernels[placement]

        kernel_val = np.reshape(kernel_val, [args.num_segments , 3,3,kernel_val.shape[-2],kernel_val.shape[-1]])
        for tt in range(args.num_segments):
            f, axarr = plt.subplots(3,3)
            for ii in range(3):
                for jj in range(3): 
                        curent_kernel = kernel_val[tt][ii][jj]
                        curent_kernel_max = curent_kernel.max()
                        curent_kernel = curent_kernel / curent_kernel_max
                        curent_kernel = (curent_kernel * 255).astype(np.uint8)
                        curent_kernel = np.array(Image.fromarray(curent_kernel).resize((224,224), resample=PIL.Image.BILINEAR))
                        curent_kernel = curent_kernel.astype(np.float32) / 255.0
                        
                        axarr[ii][jj].imshow(curent_kernel)
            
            placement = placement.replace('block','')
            folder_dir = folder + f'/viz_kernel_{placement}/'
            if not os.path.exists(folder_dir):
                os.makedirs(folder_dir)
            plt.savefig(f'{folder_dir}/mean_kernels_time_epoch_{epoch}_time_{tt}.png')  

# offsets: predicted offsets
# target_offset: distilled offsets
# input: B x TC x 224 x 224
def save_kernels(input, interm_feats, folder, name='initial', target_offset=None, target_boxes_val=None, predicted_offsets=None):
    # predicted offsets is a dict. detach when used
    predicted_boxes_dict = offsets_to_boxes(predicted_offsets,frame_size=input.shape[-1])

    places = args.place_graph.replace('layer','').split('_')
    placement_all_models = []# ['layer4', 'layer3','layer2','layer1']
    for place in places:
        layer = int(place.split('.')[0])
        block = int(place.split('.')[1])
        placement_all_models.append(f'layer{layer}_{block}')
    input_ch = 3
    if args.modality == 'gray':
        input_ch = 1
    for placement in placement_all_models:
        predicted_boxes = predicted_boxes_dict[placement]
        predicted_boxes = predicted_boxes.reshape(input.shape[0], args.num_segments, predicted_boxes.shape[1], predicted_boxes.shape[2])

        kernel_val = interm_feats[placement+'_kernels'].detach().cpu().numpy()
        video_input_val = input.detach().cpu().numpy()

        video_input_val = np.reshape(video_input_val, [args.batch_size * args.test_crops, -1, input_ch , video_input_val.shape[2],video_input_val.shape[3]])
        video_input_val = np.transpose(video_input_val, [0,1,3,4,2])
        if args.modality == 'gray':
            tmp_zeros = -1 * np.ones((video_input_val.shape[0], video_input_val.shape[1], video_input_val.shape[2], video_input_val.shape[3], 3))
            tmp_zeros[:,:,:,:,0] = video_input_val[:,:,:,:,0]
            video_input_val = tmp_zeros
      

        kernel_val = np.reshape(kernel_val, [args.batch_size * args.test_crops, -1 , kernel_val.shape[1],kernel_val.shape[2],kernel_val.shape[3],kernel_val.shape[4]])
        
        folder_dir = folder + f'/viz_kernel_{placement}/'
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)

        folder_dir = folder_dir + f'/{name}/' 
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        tt = 2

        num_rows = kernel_val.shape[2]
        save_individual_frames = True

        num_saved_videos = 2

        for video_idx in range(min(num_saved_videos, args.test_crops * args.batch_size)):
            max_t = kernel_val.shape[1]
            all_frames = []
            for tt in range(max_t):
                if 'resnet50_smt_else' in args.arch:
                    real_tt = 2*tt
                else:
                    real_tt = tt
                frame = video_input_val[video_idx,real_tt,:,:]
                if args.modality == 'RGB':
                    frame = frame * 57.375 + 114.75
                    frame = frame / 255.0
                else:
                    frame = (frame + 1.0) / 2.0

                frame_hsv = colors.rgb_to_hsv(frame)

                rgb = np.array([ [1, 0, 0 ], [0, 1, 0 ], [0, 0, 1 ]  ])
                hsv = colors.rgb_to_hsv(rgb)

                f, axarr = plt.subplots(num_rows,num_rows)
                N = kernel_val.shape[2] * kernel_val.shape[3]

                max_s = frame_hsv[:,:,1].max()
                max_v = frame_hsv[:,:,2].mean() + 0.85 * (frame_hsv[:,:,2].max() - frame_hsv[:,:,2].mean())  # poate de pus mean
                HSV_tuples = [(x*1.0/N, max_s, max_v) for x in range(N)]

                color_kernels = np.zeros( frame.shape[:2] + (3,) )
                
                cc = 0
                for ii in range(kernel_val.shape[2]):
                    for jj in range(kernel_val.shape[3]):   

                        if save_individual_frames:
                            axarr[ii][jj].imshow(kernel_val[video_idx][tt][ii][jj])
                        curent_kernel = kernel_val[video_idx][tt][ii][jj]
                        curent_kernel_max = curent_kernel.max()
                        curent_kernel = curent_kernel / curent_kernel_max
                        curent_kernel = (curent_kernel * 255).astype(np.uint8)
                        curent_kernel = np.array(Image.fromarray(curent_kernel).resize((frame.shape[0],frame.shape[1]), resample=PIL.Image.BILINEAR))
                        curent_kernel = curent_kernel.astype(np.float32) / 255.0
              
                        mask = ((curent_kernel / curent_kernel.max()) > 0.3)

                        curent_kernel = curent_kernel

                        color_kernels[:,:,0] = (1.0 - mask) * color_kernels[:,:,0] + mask * HSV_tuples[cc][0]
                        color_kernels[:,:,1] = (1.0 - mask) * color_kernels[:,:,1] + mask * HSV_tuples[cc][1]
                        color_kernels[:,:,2] = (1.0 - mask) * color_kernels[:,:,2] + mask * curent_kernel * HSV_tuples[cc][2]

                        cc += 1

                if frame.shape[2] == 1:
                    # gray to rgb
                    frame = np.tile(frame, [1,1,3])
               

                if target_offset and 'layer1' not in placement:
                    frame_distill = frame.copy()
                    for ii in range(kernel_val.shape[2] * kernel_val.shape[3]):
                        place = placement.replace('layer', '')
                        layer = int(place.split('_')[0])
                        block = int(place.split('_')[1])
                        # TODO: un-hardcodat 224 
                        h,w,dh,dw = target_offset[f'layer{layer}_block{block}'][video_idx, tt,ii] / kernel_val.shape[-1] * 224
                        frame_distill = draw_box(frame_distill, (h,w,dh,dw))
                        frame_distill = np.clip(frame_distill, 0.0, 1.0)
                rgb_color_kernels = colors.hsv_to_rgb(color_kernels)
                frame[:,:] += 1 * rgb_color_kernels#10 * kernel_sum
                frame = np.clip(frame, 0.0, 1.0)

                if save_individual_frames:
                    plt.savefig(f'{folder_dir}/kernel_{video_idx}_{tt}.png')  
                    f, axarr = plt.subplots(1,1)
                    axarr.imshow(frame) 
                    plt.savefig(f'{folder_dir}/frame_{video_idx}_{tt}.png')
                    if target_offset and 'layer1' not in placement:
                        f, axarr = plt.subplots(1,1)
                        axarr.imshow(frame_distill) 
                        plt.savefig(f'{folder_dir}/z_distill_{video_idx}_{tt}.png')                        
                plt.close(f)


                frame = (frame / frame.max() * 255.0).astype(np.uint8)
                all_frames.append(frame)
        
            clip = ImageSequenceClip(list(all_frames), fps=3)
            clip.write_gif(f'{folder_dir}/video_{video_idx}.gif', fps=3)

        plt.close('all')  


def draw_frames(input, folder, ref_point, ref_dim,  video_id, name=''):
    print(input.shape)
    video_input_val = input
    if name == 'crop':
        video_input_val = np.reshape(video_input_val, [-1, 3 , video_input_val.shape[1],video_input_val.shape[2]])
        video_input_val = np.transpose(video_input_val, [0,2,3,1])
  
    folder_dir = folder + f'/viz_kernel_tmp/'
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)

    folder_dir = folder_dir + f'/' #'./viz_kernel_center/'
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    tt = 2

    video_idx = int(video_id)
    for tt in range(16):
        frame = video_input_val[tt,:,:]
        if args.modality == 'RGB' and name == 'crop':
            frame = frame * 57.375 + 114.75
            frame = frame / 255.0
        elif name == 'crop':
            frame = (frame + 1.0) / 2.0
        else:
            frame = frame / 255.0

        frame_hsv = colors.rgb_to_hsv(frame)

        rgb = np.array([ [1, 0, 0 ], [0, 1, 0 ], [0, 0, 1 ]  ])

        frame = np.clip(frame, 0.0, 1.0)
        # draw center
        frame[ref_point[0]-3:ref_point[0]+3, ref_point[1]-3:ref_point[1]+3] = (1.0, 0.0, 0.0)
        # draw box
        left = ref_point[1] - ref_dim[1]
        right = ref_point[1] + ref_dim[1]
        up = ref_point[0] - ref_dim[0]
        down = ref_point[0] + ref_dim[0]
        frame[up-1:up+1,left:right] = (0,1,0)
        frame[down-1:down+1,left:right] = (0,1,0)
        frame[up:down,left-1:left+1] = (0,1,0)
        frame[up:down,right-1:right+1] = (0,1,0)

        f, axarr = plt.subplots(1,1)
        axarr.imshow(frame) 
        plt.savefig(f'{folder_dir}/frame_{video_idx}_{tt}_{name}.png')                        
        plt.close(f)
    plt.close('all')  



def count_params(params, contains=[''], ignores=['nonenone']):
    total_params = 0
    for name, shape in params:
        # print(f'{name} shape: {shape}')
        ok = True
        for ignore_name in ignores:
            if ignore_name in name:
                ok = False
        if not ok:
            continue
        prod = 1
        for d in shape:
            prod *= d
        selected_param = False
        for c in contains:
            if c in name:
                selected_param = True
        if selected_param:
            total_params += prod
    return total_params
def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class LearnedParamChecker():
    def __init__(self,model):
        self.model = model
        self.initial_params = self.save_initial_params()
    
    def save_initial_params(self):
        initial_params = {}
        for name, p in self.model.named_parameters():
            initial_params[name] = p.detach().cpu().numpy()
        return initial_params

    def compare_current_initial_params(self):
        for name, current_p in self.model.named_parameters():
            initial_p = self.initial_params[name]
            current_p = current_p.detach().cpu().numpy()
            diff = np.mean(np.abs(initial_p - current_p))
            print(f'params: {name} mean change : {diff}')
        


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].reshape(-1).float().sum(0)

        res.append(correct_k.mul_(100.0 / batch_size))
    return res




def adjust_kernel(pred_kernel_val, pred_boxes):
    # pred_kernel_val: (BT, 3, 3, 14, 14)
    # pred_boxes:  (BT, 9, 4) ([left_h, left_w, righ_h, right_w]))

    pred_kernel_val = np.reshape(pred_kernel_val,(pred_kernel_val.shape[0], pred_kernel_val.shape[1]*pred_kernel_val.shape[2], pred_kernel_val.shape[3], pred_kernel_val.shape[4]))
    all_adjust_kernel_boxes = np.zeros_like(pred_boxes)

    time1 = time.time()
    for b in range(pred_kernel_val.shape[0]):
        for i in range(9):
            # kernel from 14x14 ->224x224
            curent_kernel = pred_kernel_val[b,i] / pred_kernel_val[b,i].max()
            curent_kernel = (curent_kernel * 255).astype(np.uint8)

            time5 = time.time()
            curent_kernel = np.array(Image.fromarray(curent_kernel).resize((224,224), resample=PIL.Image.BILINEAR))
            time6 = time.time()
            # print(f'Resize time: {time6-time5}')
            curent_kernel = curent_kernel.astype(np.float32) / 255.0
            curent_kernel = curent_kernel / curent_kernel.sum()

            (left_h, left_w, right_h, right_w) = pred_boxes[b,i]
            h = right_h - left_h
            w = right_w - left_w
            # print(f'New frame h={h}, w={w}')
            prev_sums = curent_kernel.sum()
            prev_adjust_kernel_boxes = np.array([left_h, left_w, right_h, right_w])
            prev_dx = 0.0
            prev_dy = 0.0
            time7 = time.time()
            for j in np.arange(0.01,1, 0.01):  
                dx = j * (h/2)
                dy =  j * (w/2)
                new_left_h = int(left_h+dx)
                new_left_w = int(left_w+dy)
                new_right_h = int(right_h-dx)
                new_right_w = int(right_w-dy)
                adjust_kernel_boxes = np.array([new_left_h, new_left_w, new_right_h, new_right_w])
                crt_sums = curent_kernel[new_left_h:new_right_h+1, new_left_w:new_right_w+1].sum()
                if crt_sums < 0.90 and prev_sums >= 0.90:
                    # print(prev_dx, prev_dy, prev_sums)
                    # print(dx, dy, crt_sums)

                    # print(f'[ {left_h}, {left_w}, {right_h}, {right_w}] -> [{new_left_h}, {new_left_w}, {new_right_h}, {new_right_w}]')
                    all_adjust_kernel_boxes[b,i] = prev_adjust_kernel_boxes
                    break 

                prev_sums = crt_sums
                prev_adjust_kernel_boxes = np.array([new_left_h, new_left_w, new_right_h, new_right_w])
                prev_dx = dx
                prev_dy = dy
            time8 = time.time()
            # print(f'find adjustment: {time8-time7}') 
    time2 = time.time()
    print(f'Adjust kernel: {time2-time1}')
    return all_adjust_kernel_boxes
