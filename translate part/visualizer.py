import matplotlib.pyplot as plt
import cv2

def visualize_detection(data, img_path, output_path):
    """
    可视化检测结果
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    plt.figure(figsize=(15, 10))
    ax = plt.gca()
    
    # 绘制元件框
    for comp in data["components"]:
        x1 = int(comp["bbox"]["x1"] * width)
        y1 = int(comp["bbox"]["y1"] * height)
        x2 = int(comp["bbox"]["x2"] * width)
        y2 = int(comp["bbox"]["y2"] * height)
        
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1-5, f"{comp['type']} {comp['confidence']:.2f}",
                color='white', fontsize=8, bbox=dict(facecolor='red'))
    
    # 绘制连接线
    for conn in data["connections"]:
        path = []
        for norm_point in conn["path"]:
            x = int(norm_point[0] * width)
            y = int(norm_point[1] * height)
            path.append((x, y))
        
        xs, ys = zip(*path)
        plt.plot(xs, ys, 'g-', linewidth=1)
        plt.scatter(xs, ys, c='blue', s=20)
    
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
