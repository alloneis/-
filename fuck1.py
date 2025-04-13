# -*- coding: utf-8 -*-
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import logging
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import atexit
import platform
import math
from torchvision.models.detection.image_list import ImageList  # 新增导入
from torchvision.models.detection.faster_rcnn import TwoMLPHead

# 配置
CLASSES = ['background', 'resistor', 'capacitor', 'inductor', 'AC', 'relay',
           'voltage(1-terminal)', 'voltage(2-terminal)', 'ground', 'switch',
           'double-switch', 'transformer', 'square_wave', 'out-put', 'noise',
           'L', 'rheostat','Audio Out']
EARLY_STOP_PATIENCE = 4
DATA_PATH = r"D:\Desktop\dataset1"
BATCH_SIZE = 2
EPOCHS = 10
SYNTHETIC_SCALE = 10  # 生成15倍合成数据
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = len(CLASSES)
OUTPUT_DIR = r"D:\Desktop\OUTPUT"
# ---------------------- 数据集处理 ---------------------
class CircuitDataset(Dataset):
    # 在数据集类中添加测试方法
    def check_synthetic_data(self, save_dir="synthetic_checks"):
        os.makedirs(save_dir, exist_ok=True)
        for i in range(3):
            data = self.synthetic_data[i]
            image = data['image']
            boxes = data['boxes'].numpy()
            
            # 绘制标注框
            vis = image.copy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            
            cv2.imwrite(f"{save_dir}/syn_{i}.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    def bezier_curve(self, start, end, control_points, num_points=100):

        points = []
        for t in np.linspace(0, 1, num_points):
            x = int(self.bezier_point(t, start[0], *[p[0] for p in control_points], end[0]))
            y = int(self.bezier_point(t, start[1], *[p[1] for p in control_points], end[1]))
            points.append((x, y))
        return points

    def bezier_point(self, t, *points):
        """
        Calculate a point along a Bezier curve for a given t value.
        
        Args:
            t (float): Value between 0 and 1 representing position along the curve
            *points: Coordinate values for start, control points, and end
        
        Returns:
            float: Calculated coordinate value
        """
        n = len(points) - 1
        return sum(
            self.comb(n, i) * (1 - t)**(n - i) * t**i * points[i]
            for i in range(n + 1)
        )

    def comb(self, n, k):
        """
        Calculate the binomial coefficient (n choose k).
        
        Args:
            n (int): Total number of items
            k (int): Number of items to choose
        
        Returns:
            int: Binomial coefficient value
        """
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
    def debug_visualize(self, idx=0):
        image, target = self[idx]
        # 转换图像为numpy格式
        if isinstance(image, torch.Tensor):
            image = image.permute(1,2,0).cpu().numpy()
            image = (image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            image = image.astype(np.uint8)
        
        # 绘制真实标注框
        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(image, CLASSES[label], (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
        cv2.imwrite("dataset_debug.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print("已保存数据加载验证图：dataset_debug.jpg")
        # 在数据集中打印标注信息
        print("真实标注框数量:", len(target['boxes']))
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_paths = []
        self.annotations = []
        
        # 自动匹配图像和标注文件
        for f in os.listdir(root):
            if f.lower().endswith((".png", ".jpg")):
                base_name = os.path.splitext(f)[0]
                xml_path = os.path.join(root, f"{base_name}.xml")
                if os.path.exists(xml_path):
                    self.image_paths.append(os.path.join(root, f))
                    self.annotations.append(xml_path)
        
        # 生成合成数据
        self.synthetic_data = self.generate_synthetic_data()
        
    def generate_synthetic_data(self):
        """基于单个样本生成合成数据"""
        synthetic_data = []
        original_img = cv2.imread(self.image_paths[0])
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        canvas = np.ones((1024, 1024, 3), dtype=np.uint8) * 0x00
        # 解析原始标注
        tree = ET.parse(self.annotations[0])
        root = tree.getroot()
        
        # 提取元件信息
        components = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            components.append({
                'name': name,
                'bbox': (xmin, ymin, xmax, ymax),
                'img': original_img[ymin:ymax, xmin:xmax]
            })
        
        # 生成合成图像
        for _ in range(SYNTHETIC_SCALE):
            boxes = []
            labels = []
            
            # 随机放置元件
            for comp in components:
                # 随机缩放
                scale = random.uniform(0.8, 1.2)
                h, w = comp['img'].shape[:2]
                new_h = int(h * scale)
                new_w = int(w * scale)
                resized_img = cv2.resize(comp['img'], (new_w, new_h))
                
                # 随机位置
                max_x = 1024 - new_w
                max_y = 1024 - new_h
                if max_x < 0 or max_y < 0:
                    continue
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                
                # 粘贴元件
                canvas[y:y+new_h, x:x+new_w] = resized_img
                
                # 记录边界框
                boxes.append([x, y, x+new_w, y+new_h])
                labels.append(CLASSES.index(comp['name']))
            
            # 添加连接线
            self.add_connections(canvas, boxes)
            
            synthetic_data.append({
                'image': canvas,
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            })
        
        return synthetic_data

    def add_connections(self, canvas, boxes):
        """改进连接线生成算法"""
        # 添加曲线连接
        for _ in range(random.randint(2,5)):
            start = random.choice(boxes)
            end = random.choice(boxes)
            if start == end: continue
            
            # 贝塞尔曲线连接
            start_point = (random.randint(start[0], start[2]), 
                            random.randint(start[1], start[3]))
            end_point = (random.randint(end[0], end[2]),
                            random.randint(end[1], end[3]))
            control_points = [
                (random.randint(0,1024), random.randint(0,1024))
                for _ in range(2)
            ]
            
            # 绘制贝塞尔曲线
            points = self.bezier_curve(start_point, end_point, control_points)
            for i in range(len(points)-1):
                cv2.line(canvas, points[i], points[i+1], (0,0,0), 2)

        
    def __len__(self):
        return len(self.image_paths) + len(self.synthetic_data)

    def __getitem__(self, idx):
        if idx < len(self.image_paths):
            # 原始数据
            image = cv2.imread(self.image_paths[idx])
            # 确保三通道格式
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tree = ET.parse(self.annotations[idx])
            root = tree.getroot()
            
            boxes = []
            labels = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(CLASSES.index(name))
        else:
            # 合成数据
            data = self.synthetic_data[idx - len(self.image_paths)]
            image = data['image']
            boxes = data['boxes'].tolist()
            labels = data['labels'].tolist()

        # 数据增强
        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']

        return image, {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

# --------------------- 数据增强 ---------------------
def get_transform(train=True):
    transforms = []
    transforms.append(A.LongestMaxSize(max_size=1024))
    transforms.append(A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT))
    
    if train:
        transforms.extend([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
            A.Affine(
                translate_percent=(-0.1, 0.1),
                scale=(0.9, 1.1),
                rotate=(-5, 5),
                shear=(-5, 5),
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            # 新增随机裁剪和遮挡
            A.RandomCrop(width=800, height=800, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=64, max_width=64, p=0.5)
        ])
    
    # 确保最终输出尺寸统一为 [3, 1024, 1024]
    transforms.append(A.Resize(height=1024, width=1024))  # 添加：强制调整尺寸
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            label_fields=['labels']
        )
    )

# --------------------- 模型定义 ---------------------
def create_model():
    backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = 2048

    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32, 64, 128),),  # 更小尺寸适应元件
        aspect_ratios=((0.3, 0.5, 1.0, 2.0, 3.0),)  # 更丰富宽高比
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=NUM_CLASSES,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    # 使用标准TwoMLPHead结构
    model.roi_heads.box_head = TwoMLPHead(
        in_channels=backbone.out_channels * 7 * 7,  # 根据实际特征图尺寸调整
        representation_size=1024
    )
    return model


# --------------------- 训练流程 ---------------------
def train():
    best_val_loss = float('inf')
    dataset = CircuitDataset(DATA_PATH, get_transform(train=True))
    dataset.check_synthetic_data()
    dataset.debug_visualize()  # 添加数据验证
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0 if os.name == 'nt' else 4
    )

    model = create_model()
    model.to(DEVICE)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=3e-4,
        weight_decay=0.05
    )
    
    # 动态学习率调度
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loss_items = {
            '分类损失': 0,
            '回归损失': 0,
            'RPN分类': 0,
            'RPN回归': 0
        }
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            # 确保 images 是 Tensor 而不是 list
            # 修改：在 stack 前调整图像尺寸
            images = [img if img.shape == torch.Size([3, 1024, 1024]) else 
                      torchvision.transforms.functional.resize(img, [1024, 1024]) for img in images]
            images = torch.stack(images).to(DEVICE)  # 修改：将 images 转换为 Tensor
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            # 打印详细损失
            # 记录各项损失
            loss_items['分类损失'] += loss_dict['loss_classifier'].item()
            loss_items['回归损失'] += loss_dict['loss_box_reg'].item()
            loss_items['RPN分类'] += loss_dict['loss_objectness'].item()
            loss_items['RPN回归'] += loss_dict['loss_rpn_box_reg'].item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
        # 打印平均损失
        print(f"\nEpoch {epoch+1} 详细损失:")
        for k,v in loss_items.items():
            print(f"{k}: {v/len(train_loader):.4f}")

        avg_val_loss = total_loss / len(train_loader)  # 修改：计算平均损失
        lr_scheduler.step(avg_val_loss)  # 修改：使用平均损失更新学习率
        print(f"Epoch {epoch+1} Loss: {avg_val_loss:.4f}")
        torch.cuda.empty_cache()
        
        no_improve_epochs = 0
        if avg_val_loss < best_val_loss: 
                best_val_loss = avg_val_loss
                no_improve_epochs = 0
                torch.save(model.state_dict(), f"best_model_epoch{epoch+1}.pth")
                print(f"保存最佳模型，验证损失: {avg_val_loss:.4f}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= EARLY_STOP_PATIENCE:
                print(f"连续{EARLY_STOP_PATIENCE}个epoch未改善，提前停止")
                break

    torch.save(model.state_dict(), "circuit_detection_model.pth")
    print("训练完成，模型已保存！")

# --------------------- 可视化验证 ---------------------
def visualize_rpn_proposals(model, dataset):
    model.eval()
    image, target = dataset[0]
    with torch.no_grad():
        images = [image.to(DEVICE)]
        # 确保 images 是 Tensor 而不是 list
        images = torch.stack(images).to(DEVICE) 
        image_sizes = [img.shape[-2:] for img in images]  # 获取原始图像尺寸
        image_list = ImageList(images, image_sizes)  # 关键修复：创建 ImageList # 修改：将 images 转换为 Tensor
        # 获取 backbone 输出并转换为字典格式
        backbone_output = model.backbone(images)
        features = {'0': backbone_output}  # 修改：将 backbone 输出转换为字典
        proposals, _ = model.rpn(image_list, features)
        
        # 可视化前100个候选框   
        vis_image = image.permute(1,2,0).cpu().numpy()
        vis_image = (vis_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        vis_image = vis_image.astype(np.uint8).copy()
        
        for box in proposals[0][:100].cpu().numpy():
            x1,y1,x2,y2 = map(int, box)
            cv2.rectangle(vis_image, (x1,y1), (x2,y2), (255,0,0), 1)
        
        plt.imshow(vis_image)
        plt.show()

def visualize_results():
    dataset = CircuitDataset(DATA_PATH, get_transform(train=False))
    model = create_model()
    model.load_state_dict(torch.load("circuit_detection_model.pth"))
    model.eval()
    model.to(DEVICE)
    
    with torch.no_grad():
        for i in range(3):
            image_tensor, target= dataset[i]
            true_boxes = target['boxes'].cpu().numpy()
            true_labels = target['labels'].cpu().numpy()
            if image_tensor.ndim == 2:  # 处理灰度图
                image_tensor = image_tensor.unsqueeze(0)
            if image_tensor.shape[0] == 1:  # 单通道转三通道
                image_tensor = image_tensor.repeat(3, 1, 1)

            predictions = model([image_tensor.to(DEVICE)])  # 返回的是列表
            prediction = predictions[0]  # 获取第一个（也是唯一一个）预测结果
            pred_boxes = prediction['boxes'].cpu().numpy()
            pred_labels = prediction['labels'].cpu().numpy()
            pred_scores = prediction['scores'].cpu().numpy()
            # 转换为numpy格式
            img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            img = img_np.astype(np.uint8)
                        
            print("\n=== 诊断信息 ===")
            print(f"样本 {i} 真实标注框数量: {len(true_boxes)}")
            print(f"样本 {i} 预测框数量: {len(pred_boxes)}")
            if len(pred_boxes) > 0:
                print("预测框示例坐标:", pred_boxes[0])
                print("对应置信度:", pred_scores[0])

            # 绘制真实标注（绿色）
            for box, label in zip(true_boxes, true_labels):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, f"GT: {CLASSES[label]}", (x1, y1-30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
            # 绘制预测结果（红色）
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if score > 0.005:  # 置信度阈值
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(img, f"Pred: {CLASSES[label]} {score:.2f}", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            # 在数据集中打印标注信息
            print("真实标注框数量:", len(true_boxes))
            print("预测框数量:", len(pred_boxes))

            # 保存并显示
            output_path = os.path.join(OUTPUT_DIR, f"visualization_{i}.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            plt.figure(figsize=(12,8))
            plt.imshow(img)
            plt.show()
            print(f"可视化结果已保存至：{output_path}")
def show_exit_dialog(title, message):
    """显示退出提示框（跨平台实现）"""
    if platform.system() == 'Windows':
        # Windows系统使用原生弹窗
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, message, title, 0x40)
def exit_handler():
    """正常退出处理"""
    show_exit_dialog("程序结束", "程序已正常退出")

def exception_handler(exc_type, exc_value, exc_traceback):
    """异常处理"""
    error_msg = f"程序异常终止！\n\n错误类型: {exc_type.__name__}\n错误信息: {str(exc_value)}"
    show_exit_dialog("程序崩溃", error_msg)
    sys.__excepthook__(exc_type, exc_value, exc_traceback)  # 调用默认异常处理

def setup_exit_handling():
    """设置退出处理"""
    # 注册正常退出处理
    atexit.register(exit_handler)    
    # 设置异常钩子
    sys.excepthook = exception_handler

if __name__ == "__main__":
    setup_exit_handling()  # 必须放在所有业务代码之前
    try:
        dataset = CircuitDataset(DATA_PATH, get_transform(train=False))
        model = create_model()
        
        # 加载权重时忽略不匹配的键
        pretrained_dict = torch.load("circuit_detection_model.pth", weights_only=True)
        model_dict = model.state_dict()
        
        # 过滤掉不匹配的键
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        
        model.to(DEVICE)
        visualize_rpn_proposals(model, dataset)
        train()
        visualize_results()
    except KeyboardInterrupt:
        print("用户主动中断操作")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
