# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:14:35 2019

@author: fengjin-li
"""

from agent import Brain
from enviroment import snake
import numpy as np
import matplotlib.pyplot as plt

dingiso = Brain(n_features=12+2,use_model=False)
game_step=0
plt.ion()
plt.figure(1)
for i in range(20000):
    env = snake(4,3)
    plt.clf()
    while True:
        action = dingiso.choose_action(np.hstack((np.reshape(env.state,-1),env.body[-1])))
        state,a,r,state_=env.move(action)
        dingiso.store_transition(state,a,r,state)
        if game_step>200 and game_step%10==0:
            dingiso.learn()
#            print('learning...')
        if env.game_over is True:  
            break
        game_step+=1
        if i>10000:
            plt.imshow(env.state)
            plt.draw()#注意此函数需要调用
            plt.pause(0.5)
        
        
        
        