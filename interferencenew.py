# -*- coding: utf-8 -*-
import torch
import cv2
import os
import json
import numpy as np
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from scipy.cluster.hierarchy import fclusterdata
import matplotlib.pyplot as plt
from torchvision.models.detection.image_list import ImageList  # 新增导入
from torchvision.models.detection.faster_rcnn import TwoMLPHead
# 配置参数
CLASSES = ['background', 'resistor', 'capacitor', 'inductor', 'AC', 'relay',
           'voltage(1-terminal)', 'voltage(2-terminal)', 'ground', 'switch',
           'double-switch', 'transformer', 'square_wave', 'out-put', 'noise',
           'L', 'rheostat','Audio Out']

NUM_CLASSES = len(CLASSES)
MODEL_PATH = "circuit_detection_model.pth"
TEST_IMAGE_DIR = r"D:\test_images"
OUTPUT_DIR = r"D:\test_results"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_THRESHOLD = 0.0

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visualization"), exist_ok=True)

# 定义数据预处理
def get_transform():
    transforms = []
    transforms.append(A.LongestMaxSize(max_size=1024))
    transforms.append(A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT))
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return A.Compose(
        transforms,
    )

# 自定义RPN头（与训练时一致）
class CustomRPNHead(RPNHead):
    def __init__(self, in_channels, num_anchors):
        super().__init__(in_channels, num_anchors)
        # 修改卷积层初始化
        self.conv = torch.nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=3, stride=1, 
            padding=1, bias=True
        )
        torch.nn.init.normal_(self.conv.weight, std=0.01)
        torch.nn.init.constant_(self.conv.bias, 0)

# 加载模型（严格匹配训练配置）
def load_model(model_path):
    # 1. 构建与训练时一致的backbone
    backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)  # 修改此处
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = 2048
    
    # 2. 配置RPN参数（与训练时一致）
    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32, 64, 128),),  # 更小尺寸适应元件
        aspect_ratios=((0.3, 0.5, 1.0, 2.0, 3.0),) 
    )
    
    # 在load_model函数中添加RPN配置
    rpn_head = CustomRPNHead(
        in_channels=backbone.out_channels,
        num_anchors=anchor_generator.num_anchors_per_location()[0]
    )
    
    # 5. 构建完整模型
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=len(CLASSES),
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        box_detections_per_img=50,
        box_score_thresh=0.05,
        box_nms_thresh=0.01,
    )

    # 在推理代码中添加相同修改
    model.roi_heads.box_head = TwoMLPHead(
        in_channels=backbone.out_channels * 7 * 7,  # 根据实际特征图尺寸调整
        representation_size=1024
    )

    def feature_hook(module, input, output):
        # 获取第一个样本的特征图
        feature_maps = output[0].cpu().detach().numpy()  # [C, H, W]
        
        # 创建可视化画布
        num_channels = 32  # 显示前32个通道
        rows = 4
        cols = 8
        canvas = np.zeros((rows*64, cols*64), dtype=np.uint8)
        
        for i in range(min(num_channels, feature_maps.shape[0])):
            # 获取单个通道并归一化
            channel = feature_maps[i]
            channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-6)
            channel = (channel * 255).astype(np.uint8)
            
            # 调整大小并放置到画布
            resized = cv2.resize(channel, (64, 64))
            row = i // cols
            col = i % cols
            canvas[row*64:(row+1)*64, col*64:(col+1)*64] = resized
        
        # 保存特征图到文件，而不是显示
        feature_map_path = os.path.join(OUTPUT_DIR, "visualization", "feature_map.png")
        cv2.imwrite(feature_map_path, canvas)  # 修改此处
        print(f"Feature map saved to {feature_map_path}")

    model.backbone[0].register_forward_hook(feature_hook)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))  # 修改此处
    
    model.to(DEVICE)
    model.eval()
    return model

# 处理单张图像（保持不变）
def process_image(image_path, model, transform):
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        print(f"can't read image {image_path}")
        return None, None, None
    
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=orig_image)
    image = transformed["image"]
    
    with torch.no_grad():
        prediction = model([image.to(DEVICE)])[0]
        # 添加原始预测结果输出
    
    print("orginal prediction", prediction)
    print("预处理后图像范围:", image.min().item(), image.max().item())
    # 添加置信度分布分析
    if len(prediction["scores"]) > 0:
        print(f"max threshorld: {prediction['scores'].max().item():.4f}")
        print(f"average threshorld: {prediction['scores'].mean().item():.4f}")
    else:
        print("no prediction")

    return orig_image, prediction, orig_image.shape[:2][::-1]  # (width, height)

# 后处理函数（包含基于中心点距离的聚类）
def post_process(prediction, orig_size):
    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    
    orig_w, orig_h = orig_size
    scale = min(1024/orig_w, 1024/orig_h)
    
    # 按类别分别进行NMS
    final_boxes = []
    final_scores = []
    final_labels = []
    
    for class_id in np.unique(labels):
        # 获取当前类别的检测结果
        mask = labels == class_id
        class_boxes = boxes[mask]
        class_scores = scores[mask]
        class_labels = labels[mask]
        
        # 对每个类别单独进行NMS，使用更高的IOU阈值
        keep_indices = torchvision.ops.nms(
            torch.tensor(class_boxes), 
            torch.tensor(class_scores), 
            iou_threshold=0.01  # 提高IOU阈值
        ).numpy()
        
        final_boxes.extend(class_boxes[keep_indices])
        final_scores.extend(class_scores[keep_indices])
        final_labels.extend(class_labels[keep_indices])
    
    # 基于中心点距离进行聚类
    centers = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in final_boxes])
    # 检查 centers 是否为空或只有一个元素
    if len(centers) <= 1:
        # 如果 centers 为空或只有一个元素，直接返回结果，不进行聚类
        clustered_results = []
        for box, score, label in zip(final_boxes, final_scores, final_labels):
            xmin, ymin, xmax, ymax = box
            clustered_results.append({
                "class": CLASSES[label],
                "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                "center": [int((xmin+xmax)/2), int((ymin+ymax)/2)],
                "confidence": float(score)
            })
        return clustered_results

    # 确保 centers 是二维数组
    centers = centers.reshape(-1, 2)

    clusters = fclusterdata(centers, t=50, criterion='distance')  # t为距离阈值，可根据需要调整
    
    clustered_results = []
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        cluster_boxes = np.array(final_boxes)[cluster_mask]
        cluster_scores = np.array(final_scores)[cluster_mask]
        cluster_labels = np.array(final_labels)[cluster_mask]
        
        # 取置信度最高的框作为代表
        best_idx = np.argmax(cluster_scores)
        best_box = cluster_boxes[best_idx]
        best_score = cluster_scores[best_idx]
        best_label = cluster_labels[best_idx]
        
        # 修正坐标转换
        scale = 1024 / max(orig_h, orig_w)
        pad_x = (1024 - orig_w * scale) / 2
        pad_y = (1024 - orig_h * scale) / 2
        
        xmin = max(0, int((best_box[0] - pad_x) / scale))
        ymin = max(0, int((best_box[1] - pad_y) / scale))
        xmax = min(orig_w, int((best_box[2] - pad_x) / scale))
        ymax = min(orig_h, int((best_box[3] - pad_y) / scale))

        
        clustered_results.append({
            "class": CLASSES[best_label],
            "bbox": [xmin, ymin, xmax, ymax],
            "center": [(xmin+xmax)//2, (ymin+ymax)//2],
            "pins": get_component_pins(best_label, [xmin, ymin, xmax, ymax]),  # 新增引脚坐标计算
            "confidence": float(best_score)  # 添加置信度字段
        })
    
    return clustered_results

# 获取组件引脚坐标（示例实现）
def get_component_pins(label, bbox):
    # 根据标签和边界框计算引脚坐标
    # 这里只是一个示例实现，具体逻辑需要根据实际需求编写
    xmin, ymin, xmax, ymax = bbox
    # 修改此处：将 CLASSES.index() 的调用改为逐个检查标签是否匹配
    if label in [CLASSES.index(cls) for cls in ['background', 'resistor', 'capacitor', 'inductor', 'AC', 'relay',
                                                'voltage(1-terminal)', 'voltage(2-terminal)', 'ground', 'switch',
                                                'double-switch', 'transformer', 'square_wave', 'out-put', 'noise',
                                                'L', 'rheostat', 'Audio Out']]:
        return [(xmin, (ymin + ymax) // 2), (xmax, (ymin + ymax) // 2)]
    else:
        return []

# 保存结果（保持不变）
def save_results(image_name, results, orig_image):
    json_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(image_name)[0]}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    if orig_image is not None:
        vis_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
        for obj in results:
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(vis_image, 
                       f"{obj['class']} {obj['confidence']:.2f}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0,0,255), 1)
        
        vis_path = os.path.join(OUTPUT_DIR, "visualization", image_name)
        cv2.imwrite(vis_path, vis_image)

# 主函数
def main():
    model = load_model(MODEL_PATH)
    transform = get_transform()
    
    for img_name in os.listdir(TEST_IMAGE_DIR):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        
        img_path = os.path.join(TEST_IMAGE_DIR, img_name)
        print(f"Processing: {img_name}")
        
        orig_image, prediction, orig_size = process_image(img_path, model, transform)
        if orig_image is None:
            continue
        
        results = post_process(prediction, orig_size)
        save_results(img_name, results, orig_image)
        
        print(f"检测到 {len(results)} 个元件，结果已保存到 {OUTPUT_DIR}")
        # 新增仿真部分


if __name__ == "__main__":
    main()
    print("推理完成！所有结果已保存。")
