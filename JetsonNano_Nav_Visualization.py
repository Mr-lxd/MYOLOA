import os
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
import torch
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results

# --- 核心函数定义 ---

def get_lr_boundary_points_from_contour(contour_points, num_y_samples, height, y_buffer_percentage=0.01):
    """从轮廓点提取左右边界点"""
    if contour_points is None or len(contour_points) < 3:
        return np.array([]), np.array([])
    if contour_points.ndim == 3 and contour_points.shape[1] == 1:
        contour_points = contour_points.reshape(-1, 2)
    elif contour_points.ndim != 2 or contour_points.shape[1] != 2:
        return np.array([]), np.array([])

    min_y_contour_orig = np.min(contour_points[:, 1])
    max_y_contour_orig = np.max(contour_points[:, 1])
    contour_height = max_y_contour_orig - min_y_contour_orig

    buffer_pixels = int(contour_height * y_buffer_percentage)
    min_y_sample = min_y_contour_orig + buffer_pixels
    max_y_sample = max_y_contour_orig - buffer_pixels

    if min_y_sample >= max_y_sample:
        min_y_sample = min_y_contour_orig
        max_y_sample = max_y_contour_orig

    left_boundary_pts = []
    right_boundary_pts = []

    if max_y_sample > min_y_sample:
        sampled_y_coords = np.linspace(min_y_sample, max_y_sample, num_y_samples).astype(int)
        sampled_y_coords = np.unique(sampled_y_coords)
    else:
        if contour_height >= 1:
            sampled_y_coords = np.array([int((min_y_contour_orig + max_y_contour_orig) / 2)])
        else:
            return np.array([]), np.array([])

    for y_s in sampled_y_coords:
        x_at_y_s = []
        for i in range(len(contour_points)):
            p1 = contour_points[i]
            p2 = contour_points[(i + 1) % len(contour_points)]
            y1, y2 = p1[1], p2[1]
            x1, x2 = p1[0], p2[0]
            if (y1 <= y_s < y2) or (y2 <= y_s < y1):
                if abs(y2 - y1) > 1e-6:
                    intersect_x = x1 + (x2 - x1) * (y_s - y1) / (y2 - y1)
                    x_at_y_s.append(intersect_x)
            elif abs(y1 - y_s) < 1e-6 and abs(y2 - y_s) < 1e-6:
                x_at_y_s.extend(sorted([x1, x2]))
            elif abs(y1 - y_s) < 1e-6:
                x_at_y_s.append(x1)

        if len(x_at_y_s) >= 2:
            left_x = min(x_at_y_s)
            right_x = max(x_at_y_s)
            left_boundary_pts.append([left_x, y_s])
            right_boundary_pts.append([right_x, y_s])

    return np.array(left_boundary_pts), np.array(right_boundary_pts)

def is_detection_in_mask(detection, mask_region, overlap_threshold=0.5, min_conf=0.25):
    """检查检测框是否在掩码区域内"""
    if mask_region is None or not np.any(mask_region):
        return True

    conf = detection.get('score', 1.0)
    if conf < min_conf:
        return False

    x, y, w, h = detection['bbox']
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    x1 = max(0, min(x1, mask_region.shape[1] - 1))
    y1 = max(0, min(y1, mask_region.shape[0] - 1))
    x2 = max(0, min(x2, mask_region.shape[1] - 1))
    y2 = max(0, min(y2, mask_region.shape[0] - 1))

    if x2 <= x1 or y2 <= y1:
        return False

    bbox_mask_region = mask_region[y1:y2, x1:x2]
    bbox_area = (x2 - x1) * (y2 - y1)

    if bbox_area == 0:
        return False

    overlap_pixels = np.sum(bbox_mask_region > 0)
    overlap_ratio = overlap_pixels / bbox_area

    return overlap_ratio > overlap_threshold

def extract_detection_centerline_efficient(detections, mask_region=None, min_detections=2, overlap_threshold=0.5, min_conf=0.25):
    """从检测框提取中心线"""
    if not detections:
        return []

    center_points = []
    for det in detections:
        if mask_region is not None and np.any(mask_region):
            if not is_detection_in_mask(det, mask_region, overlap_threshold, min_conf):
                continue
        else:
            if det.get('score', 1.0) < min_conf:
                continue

        x, y, w, h = det['bbox']
        center_x = x + w / 2
        center_y = y + h / 2
        center_points.append([center_x, center_y])

    if len(center_points) < min_detections:
        return []

    center_points = np.array(center_points)
    sorted_indices = np.argsort(center_points[:, 1])
    center_points = center_points[sorted_indices]

    try:
        if len(np.unique(center_points[:, 1])) < 2:
            return [(int(p[0]), int(p[1])) for p in center_points]

        poly_coeffs = np.polyfit(center_points[:, 1], center_points[:, 0], 1)
        y_min, y_max = np.min(center_points[:, 1]), np.max(center_points[:, 1])
        y_range = np.linspace(y_min, y_max, 50)
        x_range = np.polyval(poly_coeffs, y_range)
        return list(zip(x_range.astype(int), y_range.astype(int)))
    except (np.linalg.LinAlgError, ValueError):
        return [(int(p[0]), int(p[1])) for p in center_points]

def combine_centerlines_efficient(seg_centerline_points, det_centerline_points, seg_weight=0.7, num_combined_points=50):
    """融合分割和检测导航线"""
    if not seg_centerline_points and not det_centerline_points:
        return []
    if not det_centerline_points:
        return seg_centerline_points
    if not seg_centerline_points:
        return det_centerline_points

    seg_points_np = np.array(seg_centerline_points)[np.argsort(np.array(seg_centerline_points)[:, 1])]
    det_points_np = np.array(det_centerline_points)[np.argsort(np.array(det_centerline_points)[:, 1])]

    interp_seg_x = lambda y_val: np.interp(y_val, seg_points_np[:, 1], seg_points_np[:, 0], left=seg_points_np[0, 0], right=seg_points_np[-1, 0])
    interp_det_x = lambda y_val: np.interp(y_val, det_points_np[:, 1], det_points_np[:, 0], left=det_points_np[0, 0], right=det_points_np[-1, 0])

    seg_y_min, seg_y_max = np.min(seg_points_np[:, 1]), np.max(seg_points_np[:, 1])
    det_y_min, det_y_max = np.min(det_points_np[:, 1]), np.max(det_points_np[:, 1])
    combined_y_min = min(seg_y_min, det_y_min)
    combined_y_max = max(seg_y_max, det_y_max)

    y_combined_smooth = np.linspace(combined_y_min, combined_y_max, num_combined_points)
    x_combined_smooth = []

    for y_curr in y_combined_smooth:
        x_s = interp_seg_x(y_curr)
        x_d = interp_det_x(y_curr)
        in_seg = seg_y_min <= y_curr <= seg_y_max
        in_det = det_y_min <= y_curr <= det_y_max
        w_seg = 1.0 if in_seg and not in_det else (0.0 if not in_seg and in_det else seg_weight)
        x_combined_smooth.append(w_seg * x_s + (1 - w_seg) * x_d)

    return list(zip(np.array(x_combined_smooth).astype(int), y_combined_smooth.astype(int)))

def process_and_visualize_on_image(input_image, yolo_masks_xy, yolo_masks_conf, yolo_boxes, yolo_boxes_conf, height, width,
                                   num_y_samples=50, smooth_boundaries=True, left_color=(255, 100, 100), right_color=(100, 100, 255), centerline_color=(0, 255, 0),
                                   detection_min_conf=0.25, detection_overlap_threshold=0.5, segmentation_min_conf=0.25):
    """处理图像并生成导航线"""
    output_image = input_image.copy()

    # 1. 选择最佳分割掩码（基于面积最大）
    best_mask = np.zeros((height, width), dtype=np.uint8)
    main_road_contour = None
    if yolo_masks_xy:
        max_area = -1
        for i, contour_np in enumerate(yolo_masks_xy):
            if yolo_masks_conf[i] >= segmentation_min_conf and contour_np is not None and len(contour_np) >= 3:
                area = cv2.contourArea(contour_np.astype(np.int32))
                if area > max_area:
                    max_area = area
                    main_road_contour = contour_np.astype(np.int32)
        if max_area > 0:
            cv2.drawContours(best_mask, [main_road_contour], -1, 255, thickness=cv2.FILLED)

    # 2. 提取分割导航线
    seg_centerline_points = []
    if main_road_contour is not None and len(main_road_contour) >= 3:
        left_boundary_pts, right_boundary_pts = get_lr_boundary_points_from_contour(main_road_contour, num_y_samples, height)

        if left_boundary_pts.size > 0:
            left_boundary_pts = left_boundary_pts[np.argsort(left_boundary_pts[:, 1])]
            if smooth_boundaries and len(left_boundary_pts) >= 4:
                unique_y, idx = np.unique(left_boundary_pts[:, 1], return_index=True)
                if len(unique_y) >= 4:
                    try:
                        spline = UnivariateSpline(unique_y, left_boundary_pts[idx, 0], s=len(unique_y) * 2, k=min(3, len(unique_y) - 1))
                        smooth_y = np.linspace(min(unique_y), max(unique_y), num_y_samples).astype(int)
                        smooth_x = spline(smooth_y)
                        left_boundary_pts = np.vstack((smooth_x, smooth_y)).T.astype(np.int32)
                    except Exception:
                        pass
            cv2.polylines(output_image, [left_boundary_pts], isClosed=False, color=left_color, thickness=4)

        if right_boundary_pts.size > 0:
            right_boundary_pts = right_boundary_pts[np.argsort(right_boundary_pts[:, 1])]
            if smooth_boundaries and len(right_boundary_pts) >= 4:
                unique_y, idx = np.unique(right_boundary_pts[:, 1], return_index=True)
                if len(unique_y) >= 4:
                    try:
                        spline = UnivariateSpline(unique_y, right_boundary_pts[idx, 0], s=len(unique_y) * 2, k=min(3, len(unique_y) - 1))
                        smooth_y = np.linspace(min(unique_y), max(unique_y), num_y_samples).astype(int)
                        smooth_x = spline(smooth_y)
                        right_boundary_pts = np.vstack((smooth_x, smooth_y)).T.astype(np.int32)
                    except Exception:
                        pass
            cv2.polylines(output_image, [right_boundary_pts], isClosed=False, color=right_color, thickness=4)

        if left_boundary_pts.size > 0 and right_boundary_pts.size > 0:
            left_dict = {int(pt[1]): pt[0] for pt in left_boundary_pts}
            right_dict = {int(pt[1]): pt[0] for pt in right_boundary_pts}
            common_y = sorted(set(left_dict.keys()) & set(right_dict.keys()))
            seg_centerline_points = [(int((left_dict[y] + right_dict[y]) / 2), y) for y in common_y]

    # 3. 提取检测导航线
    det_centerline_points = extract_detection_centerline_efficient(
        [{'bbox': box, 'score': conf} for box, conf in zip(yolo_boxes, yolo_boxes_conf)],
        mask_region=best_mask if np.any(best_mask) else None,
        min_detections=2,
        overlap_threshold=detection_overlap_threshold,
        min_conf=detection_min_conf
    )

    # 4. 融合导航线
    combined_centerline_points = combine_centerlines_efficient(seg_centerline_points, det_centerline_points, seg_weight=0.7, num_combined_points=num_y_samples)

    # 5. 绘制最终导航线
    if combined_centerline_points:
        points_np = np.array(combined_centerline_points, dtype=np.int32)
        if len(points_np) > 1:
            cv2.polylines(output_image, [points_np], isClosed=False, color=centerline_color, thickness=4, lineType=cv2.LINE_AA)

    return output_image, combined_centerline_points

# --- 主处理函数 ---

def run_yolo_and_postprocess():
    """运行YOLO模型并后处理图像"""
    model_path = r"/home/yy/YOLOv8-multi-task-main/runs/20250406-n-DICS_Res_add_v11_1_Seg_1_DGCST_3/weights/best.engine"
    source_path = r'/home/yy/YOLOv8-multi-task-main/test'
    output_dir = '/home/yy/YOLOv8-multi-task-main/runs/navigation_output_efficient'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        model = YOLO(model_path, task="multi")
    except Exception as e:
        return

    results_generator = model.predict(source=source_path, device=[0], imgsz=640, conf=0.25, stream=True, save=False)

    for i, results_list in enumerate(results_generator):
        if not isinstance(results_list, list):
            continue

        for idx, result in enumerate(results_list):
            if not isinstance(result, list) or len(result) < 2:
                continue

            actual_result = result[0]
            if not isinstance(actual_result, Results):
                continue

            mask_tensor = result[1]
            if not isinstance(mask_tensor, torch.Tensor):
                continue

            orig_img_bgr = actual_result.orig_img
            img_filename = actual_result.path
            if orig_img_bgr is None:
                continue

            height, width = orig_img_bgr.shape[:2]

            # 处理分割掩码
            yolo_masks_xy_all = []
            yolo_masks_conf_all = []

            if len(mask_tensor.shape) == 3:  # [num_masks, height, width]
                for k in range(mask_tensor.shape[0]):
                    mask_k = mask_tensor[k].cpu().numpy()
                    # mask_k_binary = (mask_k > 0.5).astype(np.uint8) * 255
                    mask_k_binary = mask_k.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_k_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        main_contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
                        yolo_masks_xy_all.append(main_contour)
                        yolo_masks_conf_all.append(1.0)
            elif len(mask_tensor.shape) == 2:  # [height, width]
                mask_binary = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    main_contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
                    yolo_masks_xy_all.append(main_contour)
                    yolo_masks_conf_all.append(1.0)

            # 提取检测框
            yolo_boxes_coords_all = []
            yolo_boxes_conf_all = []
            if actual_result.boxes is not None:
                boxes_xywh_abs = actual_result.boxes.xywh.cpu().numpy()
                yolo_boxes_conf_all = actual_result.boxes.conf.cpu().numpy().tolist()
                for j in range(len(boxes_xywh_abs)):
                    xc, yc, w, h_box = boxes_xywh_abs[j]
                    x_tl = xc - w / 2
                    y_tl = yc - h_box / 2
                    yolo_boxes_coords_all.append([x_tl, y_tl, w, h_box])

            # 处理并生成导航线
            processed_image, final_nav_line = process_and_visualize_on_image(
                input_image=orig_img_bgr,
                yolo_masks_xy=yolo_masks_xy_all,
                yolo_masks_conf=yolo_masks_conf_all,
                yolo_boxes=yolo_boxes_coords_all,
                yolo_boxes_conf=yolo_boxes_conf_all,
                height=height,
                width=width,
                num_y_samples=70,
                smooth_boundaries=True,
                left_color=(255, 100, 100),
                right_color=(100, 100, 255),
                centerline_color=(255, 0, 255),
                detection_min_conf=0.25,
                segmentation_min_conf=0.01,
                detection_overlap_threshold=0.5
            )

            # 保存结果
            base_filename = os.path.splitext(os.path.basename(img_filename))[0]
            output_path = os.path.join(output_dir, f"{base_filename}_nav_processed.png")
            cv2.imwrite(output_path, processed_image)

if __name__ == '__main__':
    run_yolo_and_postprocess()