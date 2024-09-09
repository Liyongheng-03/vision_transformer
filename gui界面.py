import tkinter as tk
from tkinter import filedialog
import json
from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from transformer.vision_transformer import vit_base_patch32_224,vit_base_patch16_224,VisionTransformer
import warnings
warnings.filterwarnings('ignore')
# 设置默认字体为微软雅黑
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 创建 GUI 应用程序类
def load_class_indices():
    with open("Datasets/flower/class_indices.json", "r", encoding='utf-8') as f:
        class_indict = json.load(f)
    return class_indict


class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")

        # 设置窗口的最小大小
        self.root.minsize(600, 400)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load model weights
        self.weights_path = "./results/weights/transformer/best_model_trial_39_optuna.pth" # 这个效果较差
        # self.weights_path = r"E:\研究生\transformer学习\Vit\vision_transformer\weights\model-9.pth"
        
        self.model = VisionTransformer(img_size=224,
                                       patch_size=16,
                                       embed_dim=1024,
                                       depth=4,
                                       num_heads=8,
                                       representation_size=None,
                                       num_classes=5).to(self.device)

        self.load_model_weights()

        self.data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # 加载类别索引
        self.class_indict = load_class_indices()

        # 初始化画布
        self.fig = plt.Figure(figsize=(5, 4), facecolor='lightgrey')  # 封面颜色调整为浅灰色
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')  # 隐藏坐标轴
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 创建按钮并居中
        self.load_button = tk.Button(self.root, text="Load Image", command=self.load_image, padx=40, pady=10)
        self.load_button.config(font=("Helvetica", 12, "bold"), bg='skyblue', fg='black')

        self.predict_button = tk.Button(self.root, text="Predict Image", command=self.predict_image, padx=40, pady=10)
        self.predict_button.config(font=("Helvetica", 12, "bold"), bg='lightgreen', fg='white')

        # 使用 frame 来组织按钮并居中
        self.button_frame = tk.Frame(self.root, bg='white')
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=True)  # 使用expand=True来居中按钮
        self.button_frame.config(bg='white', bd=5)

        # 按钮布局
        button_width = 10  # 按钮宽度
        self.load_button.pack(side=tk.LEFT, padx=(button_width - 2), pady=10)  # 调整padding以居中
        self.predict_button.pack(side=tk.RIGHT, padx=(button_width - 2), pady=10)


    def load_model_weights(self):
        # 加载模型权重
        self.model.load_state_dict(torch.load(self.weights_path))

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img = Image.open(file_path).convert('RGB')
            self.update_canvas(self.img)

    def update_canvas(self, img, title=""):
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.set_title(title, fontsize=14, color='black')  # 自定义标题样式
        self.fig.tight_layout()  # 调整布局以适应标题
        self.canvas.draw()

    def predict_image(self):
        if self.img is not None:
            # 预处理图像并转换为张量
            img_tensor = self.data_transform(self.img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(img_tensor)
                predict = torch.softmax(output, dim=1)
                predict_cla = torch.argmax(predict).cpu().numpy()

            # 获取预测结果的类别名称和概率
            predicted_class = self.class_indict[str(predict_cla)]
            probability = predict[0, predict_cla].item()

            # 显示预测结果
            prediction_text = f"Predicted Class: {predicted_class}\nProbability: {probability:.3f}"
            self.update_canvas(self.img, prediction_text)  # 更新画布以显示图像和预测结果
        else:
            tk.messagebox.showerror("Error", "Please load an image first.")


# 运行应用程序
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    app.root.mainloop()
