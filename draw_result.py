import matplotlib.pyplot as plt

# 训练和验证的损失和准确率数据
epochs = range(10)
train_loss = [0.795, 0.334, 0.246, 0.221, 0.201, 0.183, 0.174, 0.178, 0.177, 0.175]
valid_loss = [0.391, 0.245, 0.196, 0.172, 0.159, 0.151, 0.146, 0.144, 0.142, 0.142]
train_acc = [0.914, 0.957, 0.965, 0.958, 0.962, 0.966, 0.968, 0.964, 0.964, 0.965]
valid_acc = [0.964, 0.973, 0.977, 0.977, 0.977, 0.981, 0.981, 0.981, 0.981, 0.981]

# 绘制训练和验证损失曲线
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, valid_loss, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 绘制训练和验证准确率曲线
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
plt.plot(epochs, valid_acc, label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

