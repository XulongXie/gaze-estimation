import matplotlib.pyplot as plt
import numpy as np
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']


# 训练loss和accurancy的折线图
train_loss = np.loadtxt("./logs/Train_loss.txt")
train_accurancy = np.loadtxt("./logs/Train_accurancy.txt")

x_loss = train_loss[:, 0]
y_loss = train_loss[:, 1] / 361
x_accurancy = train_accurancy[:, 0]
y_accurancy = train_accurancy[:, 1]

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}

plt.figure(1)
plt.plot(x_loss, y_loss, 'b-', label = u'Train loss')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"loss", font1)
plt.title("Train loss", font1)
# 路径后面还要修改
plt.savefig('../../testPic/Train_loss.jpg', dpi = 300)
plt.figure(2)
plt.plot(x_accurancy, y_accurancy, 'r-', label = u'Train Accurancy')
plt.legend()
plt.margins(0)
plt.xlabel(u"epoch", font1)
plt.ylabel(u"Accurancy", font1)
plt.title("Train Accurancy", font1)
# 路径后面还要修改
plt.savefig('../../testPic/Train_accurancy.jpg', dpi = 300)
plt.show()
