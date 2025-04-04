import tensorflow as tf
import matplotlib.pyplot as plt

# 加载 MNIST 数据（包含 train 和 test）
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 构建 tf.data.Dataset 并划分验证集
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(64)
val_dataset = dataset.take(100)  # 作为验证集（小批量）
train_dataset = dataset.skip(100)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

# 构建一个简单的模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 自定义 callback
class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, model, train_data, val_data, test_data):
        super().__init__()
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

    def on_epoch_end(self, epoch, logs=None):
        train_loss = self.model.evaluate(self.train_data, verbose=0)[0]  # 只取 loss
        val_loss = self.model.evaluate(self.val_data, verbose=0)[0]
        test_loss = self.model.evaluate(self.test_data, verbose=0)[0]

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.test_losses.append(test_loss)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Test Loss={test_loss:.4f}")

    def plot_losses(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over Epochs')
        plt.grid(True)
        plt.show()

# 实例化 callback 并训练
history_callback = LossHistory(model, train_dataset, val_dataset, test_dataset)
model.fit(train_dataset, epochs=5, callbacks=[history_callback])

# 绘图
history_callback.plot_losses()
