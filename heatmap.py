import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, sys
import numpy as np

np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange # Changed from tqdm.auto
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.yolo.utils.torch_utils import intersect_dicts
from ultralytics.yolo.utils.ops import xywh2xyxy, non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


class MultiTaskActivationsAndGradients:
    def __init__(self, model, target_layers, reshape_transform, task_type='detection'):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        self.task_type = task_type

        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))
            self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients
        output.register_hook(_store_grad)

    def post_process_detection(self, result):
        if isinstance(result, (tuple, list)):
            for item in result:
                if torch.is_tensor(item): result = item; break
        if not torch.is_tensor(result): raise ValueError(f"No tensor found in detection output: {type(result)}")
        if result.dim() == 3 and result.shape[1] < result.shape[2]: result = result.permute(0, 2, 1)
        elif result.dim() == 2: result = result.unsqueeze(0)
        if result.shape[-1] < 5: raise ValueError(f"Insufficient features: {result.shape[-1]} < 5. Shape: {result.shape}")
        boxes_ = result[:, :, :4]; logits_ = result[:, :, 4:]
        max_scores = logits_[:, :, 0] if logits_.shape[-1] == 1 else logits_.max(dim=2)[0]
        indices = torch.argsort(max_scores, dim=1, descending=True)[0]
        return (logits_[0, indices], boxes_[0, indices], xywh2xyxy(boxes_[0, indices]).cpu().detach().numpy())

    def post_process_segmentation(self, result):
        if result.dim() == 4: seg_map = torch.argmax(result, dim=1)
        elif result.dim() == 3: seg_map = torch.sigmoid(result) > 0.5
        else: raise ValueError(f"Invalid segmentation output shape: {result.shape}")
        return seg_map[0]

    def __call__(self, x):
        self.gradients = []; self.activations = []
        model_output = self.model(x)

        if isinstance(model_output, (tuple, list)) and len(model_output) >= 2:
            task0_output, task1_output = model_output[0], model_output[1]
            if isinstance(task0_output, (tuple, list)) and torch.is_tensor(task1_output) and task1_output.dim() == 4:
                detection_raw_output, segmentation_raw_output = task0_output, task1_output
            elif isinstance(task1_output, (tuple, list)) and torch.is_tensor(task0_output) and task0_output.dim() == 4:
                detection_raw_output, segmentation_raw_output = task1_output, task0_output
            else:
                detection_raw_output = task0_output
                segmentation_raw_output = task1_output if len(model_output) > 1 else None

            if self.task_type == 'detection':
                det_tensor = detection_raw_output[0] if isinstance(detection_raw_output, (tuple, list)) else detection_raw_output
                if not torch.is_tensor(det_tensor): raise ValueError(f"Detection output not tensor: {type(det_tensor)}")
                if not det_tensor.requires_grad and det_tensor.is_leaf: det_tensor.requires_grad_(True)
                logits, boxes, _ = self.post_process_detection(det_tensor)
                return [[logits, boxes]]
            elif self.task_type == 'segmentation':
                if not torch.is_tensor(segmentation_raw_output): raise ValueError(f"Seg output not tensor: {type(segmentation_raw_output)}")
                if not segmentation_raw_output.requires_grad and segmentation_raw_output.is_leaf:
                    segmentation_raw_output.requires_grad_(True)
                return segmentation_raw_output
        else:
            if torch.is_tensor(model_output):
                if not model_output.requires_grad and model_output.is_leaf: model_output.requires_grad_(True)
                if self.task_type == 'detection':
                    logits, boxes, _ = self.post_process_detection(model_output)
                    return [[logits, boxes]]
                elif self.task_type == 'segmentation': return model_output
            raise ValueError("Expected multi-task output or single tensor.")

    def release(self):
        for handle in self.handles: handle.remove()


class yolov8_detection_target(torch.nn.Module):
    def __init__(self, output_type, conf, ratio):
        super().__init__(); self.output_type, self.conf, self.ratio = output_type, conf, ratio

    def forward(self, data):
        logits, boxes = data
        targets = []
        num_to_consider = max(1, int(logits.size(0) * self.ratio)) if logits.size(0) > 0 else 0
        for i in range(num_to_consider):
            score = logits[i].max()
            if score.item() < self.conf: break
            if self.output_type in ['class', 'all']: targets.append(score)
            if self.output_type in ['box', 'all']: targets.extend(boxes[i])
        if not targets:
            device = logits.device if logits.numel() > 0 else (boxes.device if boxes.numel() > 0 else torch.device('cpu'))
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(targets).sum()


class yolov8_segmentation_target(torch.nn.Module):
    def __init__(self, conf_threshold=0.5, target_class_index=None):
        super().__init__(); self.conf_threshold, self.target_class_index = conf_threshold, target_class_index

    def forward(self, raw_seg_logits):
        logits_b0 = raw_seg_logits[1] # target_class_index
        if logits_b0.dim() == 2:
            return logits_b0.sum()
        elif logits_b0.dim() == 3:
            if self.target_class_index is not None:
                class_logits = logits_b0[self.target_class_index]
            elif logits_b0.shape[0] == 1:
                class_logits = logits_b0[0]
            elif logits_b0.shape[0] == 2:
                class_logits = logits_b0[1]
            else:
                class_logits = torch.max(logits_b0, dim=0)[0]
            return class_logits.sum()
        return torch.tensor(0.0, device=raw_seg_logits.device, requires_grad=True)


class yolov8_multitask_heatmap:
    def __init__(self, weight, device, method, detection_layers, segmentation_layers,
                 backward_type, conf_threshold, ratio, show_box, renormalize):
        device = torch.device(device)
        ckpt = torch.load(weight, map_location=device)
        model_names = ckpt['model'].names if hasattr(ckpt.get('model'), 'names') else {} # Added safety for names
        model = attempt_load_weights(weight, device=device, fuse=False)
        # model.info() # Optional: for debugging model structure

        for p in model.parameters(): p.requires_grad_(True)
        model.eval()

        self.detection_target = yolov8_detection_target(backward_type, conf_threshold, ratio)
        self.segmentation_target = yolov8_segmentation_target(conf_threshold)

        self.specified_detection_layers = detection_layers # Store for filename
        self.specified_segmentation_layers = segmentation_layers # Store for filename

        if detection_layers and len(detection_layers) > 0 :
            det_target_layers_modules = [model.model[l] for l in detection_layers]
            self.detection_method = eval(method)(model, det_target_layers_modules)
            self.detection_method.activations_and_grads = MultiTaskActivationsAndGradients(
                model, det_target_layers_modules, None, task_type='detection')
        else:
            self.detection_method = None

        if segmentation_layers and len(segmentation_layers) > 0:
            seg_target_layers_modules = [model.model[l] for l in segmentation_layers]
            self.segmentation_method = eval(method)(model, seg_target_layers_modules)
            self.segmentation_method.activations_and_grads = MultiTaskActivationsAndGradients(
                model, seg_target_layers_modules, None, task_type='segmentation')
        else:
            self.segmentation_method = None

        colors = np.random.uniform(0, 255, size=(len(model_names) if model_names else 80, 3)).astype(np.int32)
        self.__dict__.update(locals())

    def post_process_detection(self, result):
        if isinstance(result, (tuple, list)):
            for item in result:
                if torch.is_tensor(item): result = item; break
        if not torch.is_tensor(result): raise ValueError(f"No tensor in det output: {type(result)}")

        if result.dim() == 3 and result.shape[1] < result.shape[2]: result = result.permute(0, 2, 1)
        elif result.dim() == 2: result = result.unsqueeze(0)
        if result.shape[-1] < 5: raise ValueError(f"Insufficient features: {result.shape[-1]} < 5. Shape: {result.shape}")

        boxes_ = result[:, :, :4]; logits_ = result[:, :, 4:]
        max_scores = logits_[:, :, 0] if logits_.shape[-1] == 1 else logits_.max(dim=2)[0]
        indices = torch.argsort(max_scores, dim=1, descending=True)[0]
        return (logits_[0, indices], boxes_[0, indices], xywh2xyxy(boxes_[0, indices]).cpu().detach().numpy())

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = map(int, list(box))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2, lineType=cv2.LINE_AA)
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        renormalized_cam = np.zeros_like(grayscale_cam, dtype=np.float32)
        for x1, y1, x2, y2 in boxes.astype(np.int32):
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            if x1 < x2 and y1 < y2:
                 patch = grayscale_cam[y1:y2, x1:x2]
                 if patch.size > 0:
                    renormalized_cam[y1:y2, x1:x2] = scale_cam_image(patch.copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        return show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)

    def process_detection(self, img_path, save_path):
        if not self.detection_method:
            # print("Skipping detection heatmap: No detection layers specified.") # Already handled in __call__
            return False
        img = cv2.imread(img_path)
        img_letterboxed = letterbox(img)[0]
        img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
        img_float = np.float32(img_rgb) / 255.0
        tensor = torch.from_numpy(np.transpose(img_float, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)

        try:
            grayscale_cam = self.detection_method(tensor, [self.detection_target])
            if grayscale_cam is None or grayscale_cam.size == 0:
                print(f"Detection heatmap generation failed: grayscale_cam is empty for {os.path.basename(img_path)}")
                return False
        except Exception as e:
            print(f"Detection heatmap generation failed for {os.path.basename(img_path)}: {e}")
            # import traceback; traceback.print_exc() # Can be verbose
            return False

        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        pred_boxes_np = None
        with torch.no_grad():
            model_output = self.model(tensor)
            det_output_tensor = None
            if isinstance(model_output, (list, tuple)) and len(model_output) > 0:
                det_output = model_output[0]
                if isinstance(det_output, (list, tuple)) and len(det_output) > 0 and torch.is_tensor(det_output[0]):
                    det_output_tensor = det_output[0]
                elif torch.is_tensor(det_output):
                    det_output_tensor = det_output
                else:
                    found_det_tensor = False
                    for item_mo in model_output: # Renamed item to item_mo to avoid conflict
                        if isinstance(item_mo, (list,tuple)) and len(item_mo)>0 and torch.is_tensor(item_mo[0]) and item_mo[0].ndim >=2:
                             if item_mo[0].shape[-1] > 4:
                                det_output_tensor = item_mo[0]; found_det_tensor = True; break
                        elif torch.is_tensor(item_mo) and item_mo.ndim >=2:
                            if item_mo.shape[-1] > 4 or (item_mo.ndim==3 and item_mo.shape[1] > 4):
                                det_output_tensor = item_mo; found_det_tensor = True; break
                    if not found_det_tensor: print(f"Could not reliably identify detection tensor for {os.path.basename(img_path)}.")
            elif torch.is_tensor(model_output):
                det_output_tensor = model_output
            else: print(f"Model output is not a list/tuple or tensor for {os.path.basename(img_path)}.")

            if det_output_tensor is not None:
                try:
                    processed_logits, processed_boxes_xywh, pred_boxes_xyxy_np = self.post_process_detection(det_output_tensor.clone().detach())
                    pred_boxes_np = pred_boxes_xyxy_np
                    if self.show_box and pred_boxes_np is not None and pred_boxes_np.shape[0] > 0:
                        num_preds_to_draw = pred_boxes_np.shape[0]
                        scores, class_indices = (torch.sigmoid(processed_logits[:num_preds_to_draw, 0]), torch.zeros_like(processed_logits[:num_preds_to_draw, 0], dtype=torch.long)) if processed_logits.shape[-1] == 1 else torch.softmax(processed_logits[:num_preds_to_draw], dim=-1).max(dim=-1)
                        for i in range(min(num_preds_to_draw, 10)):
                            if scores[i] >= self.conf_threshold:
                                box = pred_boxes_np[i, :4]
                                class_id = class_indices[i].item()
                                class_name = self.model_names.get(class_id, f"cls_{class_id}")
                                label = f'{class_name} {scores[i]:.2f}'
                                cam_image = self.draw_detections(box, self.colors[class_id % len(self.colors)], label, cam_image)
                    if self.renormalize and pred_boxes_np is not None and pred_boxes_np.shape[0] > 0:
                         cam_image = self.renormalize_cam_in_bounding_boxes(pred_boxes_np[:, :4], img_float, grayscale_cam)
                except Exception as e_postproc: print(f"Error during post-processing detections for drawing {os.path.basename(img_path)}: {e_postproc}")
        print(
            f"Grayscale CAM stats for {os.path.basename(img_path)} (Det L{'_'.join(map(str, self.specified_detection_layers))}): min={grayscale_cam.min()}, max={grayscale_cam.max()}, mean={grayscale_cam.mean()}")
        Image.fromarray(cam_image).save(save_path)
        return True

    def process_segmentation(self, img_path, save_path):
        if not self.segmentation_method:
            # print("Skipping segmentation heatmap: No segmentation layers specified.")
            return False
        img = cv2.imread(img_path)
        img_letterboxed = letterbox(img)[0]
        img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
        img_float = np.float32(img_rgb) / 255.0
        tensor = torch.from_numpy(np.transpose(img_float, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)

        try:
            grayscale_cam = self.segmentation_method(tensor, [self.segmentation_target])
            if grayscale_cam is None or grayscale_cam.size == 0:
                print(f"Segmentation heatmap generation failed: grayscale_cam is empty for {os.path.basename(img_path)}")
                return False
        except Exception as e:
            print(f"Segmentation heatmap generation failed for {os.path.basename(img_path)}: {e}")
            # import traceback; traceback.print_exc()
            return False

        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        Image.fromarray(cam_image).save(save_path)
        return True

    def __call__(self, img_path, save_base_path):
        det_layers_str = "_".join(map(str, self.specified_detection_layers)) if self.specified_detection_layers else "None"
        seg_layers_str = "_".join(map(str, self.specified_segmentation_layers)) if self.specified_segmentation_layers else "None"

        detection_save_dir_name = f'detection_heatmaps_L{det_layers_str}'
        segmentation_save_dir_name = f'segmentation_heatmaps_L{seg_layers_str}'

        detection_save_path = os.path.join(save_base_path, detection_save_dir_name)
        segmentation_save_path = os.path.join(save_base_path, segmentation_save_dir_name)

        if self.detection_method: os.makedirs(detection_save_path, exist_ok=True)
        if self.segmentation_method: os.makedirs(segmentation_save_path, exist_ok=True)

        if os.path.isdir(img_path):
            img_files = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            for img_file_idx in trange(len(img_files), desc=f"Processing Imgs (DetL:{det_layers_str}, SegL:{seg_layers_str})"):
                img_name = img_files[img_file_idx]
                # print(f"\nProcessing {img_name}...") # tqdm provides progress
                img_full_path = os.path.join(img_path, img_name)

                if self.detection_method:
                    detection_save_file = os.path.join(detection_save_path, f"det_{img_name}")
                    det_success = self.process_detection(img_full_path, detection_save_file)
                    if not det_success: print(f"✗ Detection heatmap failed: {img_name}")
                    # else: print(f"✓ Detection heatmap saved: {detection_save_file}") # Can be too verbose with tqdm

                if self.segmentation_method:
                    segmentation_save_file = os.path.join(segmentation_save_path, f"seg_{img_name}")
                    seg_success = self.process_segmentation(img_full_path, segmentation_save_file)
                    if not seg_success: print(f"✗ Segmentation heatmap failed: {img_name}")
                    # else: print(f"✓ Segmentation heatmap saved: {segmentation_save_file}")
        else:
            img_name = os.path.basename(img_path)
            print(f"Processing single image: {img_name} (DetL:{det_layers_str}, SegL:{seg_layers_str})...")
            if self.detection_method:
                detection_save_file = os.path.join(detection_save_path, f"det_{img_name}")
                det_success = self.process_detection(img_path, detection_save_file)
                print(f"✓ Detection heatmap saved: {detection_save_file}" if det_success else f"✗ Detection heatmap failed: {img_name}")

            if self.segmentation_method:
                segmentation_save_file = os.path.join(segmentation_save_path, f"seg_{img_name}")
                seg_success = self.process_segmentation(img_path, segmentation_save_file)
                print(f"✓ Segmentation heatmap saved: {segmentation_save_file}" if seg_success else f"✗ Segmentation heatmap failed: {img_name}")
        print(f"Finished processing. Results in: {save_base_path}")


def get_params():
    params = {
        'weight': 'runs/20250406-n-DICS_Res_add_v11_1_Seg_1_DGCST_3/weights/best.pt',
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'method': 'GradCAM',
        'detection_layers': [],
        'segmentation_layers': [28],
        'backward_type': 'all',
        'conf_threshold': 0.25,
        'ratio': 0.02,
        'show_box': False,
        'renormalize': False
    }
    return params

if __name__ == '__main__':
    config_params = get_params()
    if not (config_params.get('detection_layers') and len(config_params['detection_layers']) > 0) and \
       not (config_params.get('segmentation_layers') and len(config_params['segmentation_layers']) > 0) :
        print("Error: No layers specified for either detection or segmentation heatmaps. Exiting.")
        sys.exit(1)

    heatmap_generator = yolov8_multitask_heatmap(
        weight=config_params['weight'],
        device=config_params['device'],
        method=config_params['method'],
        detection_layers=config_params['detection_layers'],
        segmentation_layers=config_params['segmentation_layers'],
        backward_type=config_params['backward_type'],
        conf_threshold=config_params['conf_threshold'],
        ratio=config_params['ratio'],
        show_box=config_params['show_box'],
        renormalize=config_params['renormalize']
    )

    heatmap_generator(r'D:/navigationData/ImagesForYOLOv8Mutil_3/images/test', 'heatmap/MYOLOA')