#方法1：来自官网，两条曲线分开绘制
#训练历史可视化
import matplotlib.pyplot as plt

history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#方法2：两条曲线绘制在一起
    def draw_acc_loss(history):
        acc = history.history['accuracy']
        loss = history.history['loss']
        epochs = range(1, len(acc) + 1)
        plt.title('Accuracy and Loss')
        plt.plot(epochs, acc, 'red', label='Training acc')
        plt.plot(epochs, loss, 'blue', label='Validation loss')
        plt.legend()
        plt.show()
