# -*- coding : utf-8-*-
import json
# 导入套接字模块
import socket
# 导入线程模块
import threading

import numpy as np
import visdom

from AGENT import AGENT
from ENV.ENV import ENV
from TRAIN import train, sampling
from UTILS.parameters import define_parameters


# 定义个函数,使其专门重复处理客户的请求数据（也就是重复接受一个用户的消息并且重复回答，直到用户选择下线）
def dispose_client_request(tcp_client_obj_):
    # 5 循环接收和发送数据
    data_str = ''
    while True:
        re = tcp_client_obj_.recv(1024).decode()
        data_str = data_str + re
        if "stop" in data_str:
            break
    data = json.loads(data_str)
    # 6 接收数据解析
    # 6.1 FW-UAV 轨迹
    fw_uav_center = np.array(data['fw_uav_center'])
    fw_uav_radius = np.array(data['fw_uav_radius'])
    fw_uav_radius_max = np.array(data['fw_uav_radius_max'])
    fw_uav_radius_min = np.array(data['fw_uav_radius_min'])
    fw_uav_speed = np.array(data['fw_uav_speed'])
    # 6.2 RW-UAVs 相关数据
    num_rw_uav = np.array(data['num_rw_uav'])
    rw_uav_tra = np.array(data['rw_uav_tra'])
    # 6.3 通信参数
    num_slot = np.array(data['num_slot'])
    length_slot = np.array(data['length_slot'])
    p = np.array(data['p'])
    bw = np.array(data['bw'])
    bw_ul = np.array(data['bw_ul'])
    noise_den = np.array(data['noise_den'])
    beta = np.array(data['beta'])
    rw_uav_p = np.array(data['rw_uav_p'])
    # 6.4 DRL 参数
    episode_max = np.array(data['episode_max'])
    step_max = np.array(data['step_max'])
    dim_state = np.array(data['dim_state'])
    dim_action = np.array(data['dim_action'])

    # 参数定义
    par = define_parameters()
    par.fw_uav_center = fw_uav_center
    par.fw_uav_radius = fw_uav_radius
    par.fw_uav_radius_max = fw_uav_radius_max
    par.fw_uav_radius_min = fw_uav_radius_min
    par.fw_uav_speed = fw_uav_speed
    par.num_rw_uav = num_rw_uav
    par.rw_uav_tra = rw_uav_tra
    par.num_slot = num_slot
    par.length_slot = length_slot
    par.p = p
    par.bw = bw
    par.rw_uav_p = rw_uav_p
    par.noise_den = noise_den
    par.beta = beta
    par.episode_max = episode_max
    par.step_max = step_max
    par.dim_state = dim_state
    par.dim_action = dim_action
    par.bw_ul = bw_ul
    par.mean = 1
    par.std = 1
    viz = None
    if par.visdom_flag:
        viz = visdom.Visdom()
        viz.close()
    # 2. Create the environment
    env = ENV(par)
    # 3. Create the agent
    index_loss_ac = 0
    index_loss_cr = 0
    # 创建智能体
    # 采样定归一化的均值与方差
    agent_test = AGENT.AGENT(par, env, viz, index_loss_ac, index_loss_cr)
    mean, std = sampling(agent_test, env, par.batch_size)
    par.mean = mean
    par.std = std

    agent = AGENT.AGENT(par, env, viz, index_loss_ac, index_loss_cr)
    action_res, _, reward = train(par, agent, env, viz)

    # 7 结果打包
    action_res = action_res.tolist()
    # rate_res = rate_res.tolist()
    res = json.dumps([reward, action_res])
    tcp_client_obj_.sendall(res.encode('utf-8'))
    tcp_client_obj_.close()
    print("关闭套接字")


if __name__ == '__main__':
    client_id = 0
    # 1 创建服务端套接字对象
    tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 设置端口复用，使程序退出后端口马上释放
    tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
    # 2 绑定端口
    tcp_server.bind(("127.0.0.1", 12345))
    # 3 设置监听
    tcp_server.listen(128)
    # 4 循环等待客户端连接请求（也就是最多可以同时有128个用户连接到服务器进行通信）
    while True:
        tcp_client_obj, tcp_client_address = tcp_server.accept()
        print("有客户端接入:", tcp_client_address)
        # 创建多线程对象
        thread = (threading.Thread(target=dispose_client_request, args=(tcp_client_obj,)))
        # 设置守护主线程  即如果主线程结束了 那子线程中也都销毁了  防止主线程无法退出
        thread.setDaemon(True)
        # 启动子线程对象
        thread.start()
        client_id += 1
