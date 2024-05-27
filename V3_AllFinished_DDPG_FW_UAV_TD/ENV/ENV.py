import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull


class ENV:
    def __init__(self, par):
        self.par = par
        self.radius_fw_uav = par.fw_uav_radius
        self.center_fw_uav = par.fw_uav_center
        self.start_fw_uav = [self.center_fw_uav[0] - self.radius_fw_uav, self.center_fw_uav[1]]
        self.num_slot = self.par.num_slot
        self.time_interval = par.length_slot
        self.tra_rw_uav = self.par.rw_uav_tra
        self.fw_uav_speed = self.par.fw_uav_speed
        self.tra_fw_uav = self.transform_fw_uav_trajectory(self.radius_fw_uav, self.center_fw_uav, self.fw_uav_speed)
        self.dim_action = 3
        self.dim_state = 5 + self.par.num_rw_uav
        self.center_now = self.center_fw_uav
        self.radius_now = self.radius_fw_uav
        self.range_center_x, self.range_center_y = self.obtain_action_space()

    def reset(self):
        # State->[radius, center_x, center_y, start_x, start_y, sum rate with time slot between FW-UAV and RW-UAVs]
        # reset(): Update the trajectory of the FW-UAV to the init trajectory include radius and the center
        self.center_now = self.par.fw_uav_center
        self.radius_now = self.par.fw_uav_radius
        rate = self.calculate_rate_fw_rw_uav(self.tra_fw_uav)
        obs_1 = [self.radius_fw_uav, self.center_fw_uav[0], self.center_fw_uav[1], self.start_fw_uav[0],
                 self.start_fw_uav[1]]
        obs_init = np.concatenate((obs_1, rate))
        return obs_init

    def step(self, action, index_step):
        done = False
        # print(action)
        # center_dir = action[0]
        # center_dis = action[1]
        radius = action[2]
        # Calculate the new center
        center_new = np.zeros(2)
        center_new[0] = action[0]
        center_new[1] = action[1]
        # center_new[0] = self.center_now[0] + center_dis * np.cos(center_dir)
        # center_new[1] = self.center_now[1] + center_dis * np.sin(center_dir)
        self.center_now = center_new
        self.radius_now = radius
        tra_fw_uav = self.transform_fw_uav_trajectory(radius, center_new, 50)
        start_point = self.calculate_start_point(radius, center_new)
        rate = self.calculate_rate_fw_rw_uav(tra_fw_uav)
        if index_step >= self.par.step_max:
            done = True
        obs_1 = [self.radius_now, self.center_now[0], self.center_now[1], start_point[0], start_point[1]]
        obs_next = np.concatenate((obs_1, rate))
        reward = np.sum(rate) / (10 ** 6)
        return obs_next, reward, np.sum(rate), done

    def generate_rw_uav_trajectory(self):
        # The trajectory of RW-UAVs, consisting: [time slot index, UAV ID, x, y, z]
        rw_uav_trajectory = np.zeros((self.par.num_slot, self.par.num_rw_uav, 3))
        for index_slot in range(self.par.num_slot):
            for index_rw_uav in range(self.par.num_rw_uav):
                rw_uav_trajectory[index_slot, index_rw_uav, 0] = (np.random.uniform(low=index_slot * 100 - 50,
                                                                                    high=index_slot * 100 + 50)
                                                                  )
                rw_uav_trajectory[index_slot, index_rw_uav, 1] = (np.random.uniform(low=index_slot * 100 - 50,
                                                                                    high=index_slot * 100 + 50)
                                                                  ) + index_rw_uav * 800
                rw_uav_trajectory[index_slot, index_rw_uav, 2] = (index_rw_uav + 1) * 50
        return rw_uav_trajectory

    def transform_fw_uav_trajectory(self, radius, center, speed):
        tra = np.zeros((self.num_slot, 3))
        tra[0, 0] = center[0] - radius
        tra[0, 1] = center[1]
        tra[0, 2] = 1000
        # The length of the arc in each time slot, Units:rad
        theta_each_slot = (speed * self.time_interval) / radius
        for index_slot in range(1, self.par.num_slot):
            tra[index_slot, 0] = center[0] - radius * np.cos(index_slot * theta_each_slot)
            tra[index_slot, 1] = center[1] - radius * np.sin(index_slot * theta_each_slot)
            tra[index_slot, 2] = 1000
        return tra

    def calculate_rate_fw_rw_uav(self, tra_rw_uav):
        rate_dl = np.zeros(self.par.num_rw_uav)
        rate_ul = np.zeros(self.par.num_rw_uav)
        for index_uav in range(self.par.num_rw_uav):
            dis = np.sqrt(
                np.sum((tra_rw_uav[:, 0:2] - np.reshape(self.tra_rw_uav[:, index_uav, 0:2], (self.num_slot, 2))) ** 2,
                       1) + (tra_rw_uav[:, 2] - np.reshape(self.tra_rw_uav[:, index_uav, 2], (1, -1))) ** 2)
            gain = self.par.beta / (dis ** 2)
            sinr_dl = np.reshape(self.par.p[:, index_uav], (1, -1)) * gain / (
                    np.reshape(self.par.bw[:, index_uav], (1, -1)) * self.par.noise_den)
            a = np.reshape(self.par.bw[:, index_uav], (1, -1)) * np.log2(1 + sinr_dl)
            rate_dl[index_uav] = np.sum(a)
            sinr_ul = self.par.rw_uav_p * gain / (
                    np.reshape(self.par.bw_ul[:, index_uav], (1, -1)) * self.par.noise_den)
            a = np.reshape(self.par.bw_ul[:, index_uav], (1, -1)) * np.log2(1 + sinr_ul)
            rate_ul[index_uav] = np.sum(a)
        return rate_dl + rate_ul

    @staticmethod
    def calculate_start_point(radius, center):
        temp = [center[0] - radius, center[1]]
        return np.array(temp)

    def obtain_action_space(self):
        point = np.zeros((self.num_slot * self.par.num_rw_uav, 2))
        index = 0
        for index_slot in range(self.num_slot):
            for index_uav in range(self.par.num_rw_uav):
                point[index][0] = self.tra_rw_uav[index_slot][index_uav][0]
                point[index][1] = self.tra_rw_uav[index_slot][index_uav][1]
                index += 1
        x_max = np.max(point[:, 0])
        x_min = np.min(point[:, 0])
        y_max = np.max(point[:, 1])
        y_min = np.min(point[:, 1])
        x_range = [x_min, x_max]
        y_range = [y_min, y_max]
        # plt.figure()
        # plt.scatter(point[:, 0], point[:, 1])
        # x_axis = np.linspace(x_min, x_max)
        # y_axis = np.linspace(y_min, y_max)
        # plt.plot(x_axis, y_max * np.ones(len(x_axis)))
        # plt.plot(x_axis, y_min * np.ones(len(x_axis)))
        # plt.plot(x_max * np.ones(len(x_axis)), y_axis)
        # plt.plot(x_min * np.ones(len(x_axis)), y_axis)
        # plt.show()
        # print(x_range)
        # print(y_range)
        return x_range, y_range
