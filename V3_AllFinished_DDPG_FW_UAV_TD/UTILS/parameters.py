"""
Author   : Bao-lin Yin
Data     : 2.27 2024
Version  : V1.0
Function : Defining the parameters.
"""
import argparse


def define_parameters():
    para = argparse.ArgumentParser("The parameters for solving the trajectory design of FW-UAV")
    # The parameters for the DDPG
    para.add_argument('--lr_ac', type=float, default=1e-3, help='The learning rate of actor')
    para.add_argument('--lr_cr', type=float, default=1e-3, help='The learning rate of critic')
    para.add_argument('--gamma', type=float, default=0.9, help='The discount factor')
    para.add_argument('--epsilon', type=float, default=0.9, help='The greedy factor')
    para.add_argument('--memory_capacity', type=int, default=10000, help='The size of the memory')
    para.add_argument('--batch_size', type=int, default=128, help='The batch size')
    para.add_argument('--tau', type=float, default=0.008, help='The tau')

    # The parameters for the plot
    para.add_argument('--visdom_flag', type=bool, default=True, help='visdom is enabled')

    args = para.parse_args()
    return args
