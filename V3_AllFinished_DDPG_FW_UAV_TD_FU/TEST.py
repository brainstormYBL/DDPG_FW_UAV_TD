import numpy as np

from ENV import ENV
from UTILS.parameters import define_parameters
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Test the environment
def circle_array(xc, yc, r, start, end):
    # 根据圆心、半径、起始角度、结束角度，生成圆弧的数据点
    phi1 = start * np.pi / 180.0
    phi2 = end * np.pi / 180.0
    d_phi = (phi2 - phi1) / np.ceil(200 * np.pi * r * (phi2 - phi1))  # 根据圆弧周长设置等间距
    array = np.arange(phi1, phi2, d_phi)
    array = np.append(array, array[-1] + d_phi)  # array在结尾等距增加一个元素
    return xc + r * np.cos(array), yc + r * np.sin(array)


par = define_parameters()
env = ENV.ENV(par)

# 创建三维图像对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制曲线
for index_uav in range(par.num_rw_uav):
    ax.plot(env.tra_rw_uav[:, index_uav, 0], env.tra_rw_uav[:, index_uav, 1], env.tra_rw_uav[:, index_uav, 2])

ax.plot(env.tra_fw_uav[:, 0], env.tra_rw_uav[:, 1], env.tra_fw_uav[:, 2])

# 设置轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图像
plt.show()

plt.figure()
for index_uav in range(par.num_rw_uav):
    plt.plot(env.tra_rw_uav[:, index_uav, 0], env.tra_rw_uav[:, index_uav, 1])
plt.show()

X1, Y1 = circle_array(env.center_fw_uav[0], env.center_fw_uav[1],env.radius_fw_uav,0,360)
plt.figure()
plt.plot(X1, Y1)
plt.scatter(env.tra_fw_uav[:, 0], env.tra_fw_uav[:, 1])
plt.show()

par.p = (10 / par.num_rw_uav) * np.ones((par.num_slot, par.num_rw_uav))
par.bw = (40 / par.num_rw_uav) * np.ones((par.num_slot, par.num_rw_uav))
par.beta = 1
par.noise_den = 0.01
obs_init = env.reset()
point = env.obtain_action_space()
