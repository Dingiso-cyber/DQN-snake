# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:20:42 2019

@author: fengjin-li
"""
import keras
import numpy as np
from keras.layers import Input,Dense
class Brain():
    
    def __init__(self,
                 n_features=100,
                 n_actions=4,
                 learning_rate=0.1,
                 memory_size=5000,
                 reward_decay=0.9,
                 batch_size = 64,
                 epsilon=0.9,
                 epsilon_decay=0.99,
                 use_model=True,
                 ):
        self.all=[]
        self.use_model=use_model
        self.gamma = reward_decay
        self.n_features = n_features
        self.n_actions = n_actions
        self.learning_rate  = learning_rate
        self.memory_size = memory_size
        self.reward_decay = reward_decay
        self.batch_size = batch_size
        self.build_net()
        self.memory = np.zeros([self.memory_size,self.n_features*2+2])
        self.epsilon=0.9
        self.epsilon_decay=0.99
        self.learn_step_count = 0
        self.replace_iter = 100

        
        
    def build_net(self):
        if self.use_model:
            self.model1=keras.models.load_model('model1.h5')
            self.model2=keras.models.load_model('model1.h5')
        else:
        #build eval_net
            eval_inputs = Input(shape = (self.n_features,))
            x = Dense(50,activation='relu')(eval_inputs)
            x = Dense(50,activation='relu')(x)
            x = Dense(50,activation='relu')(x)
            x = Dense(50,activation='relu')(x)
            x = Dense(10,activation='relu')(x)
            self.q_eval = Dense(self.n_actions)(x)
            #build q_target_net
            target_inputs = Input(shape=(self.n_features,))
            x = Dense(50,activation='relu')(target_inputs)
            x = Dense(50,activation='relu')(x)
            x = Dense(50,activation='relu')(x)
            x = Dense(50,activation='relu')(x)
            x = Dense(10,activation='relu')(x)
            self.q_target = Dense(self.n_actions)(x)
            self.model1 = keras.models.Model(eval_inputs,self.q_eval)
            self.model2 = keras.models.Model(target_inputs,self.q_target)
            self.model1.compile(optimizer='rmsprop',loss='mse')
            self.model2.compile(optimizer='rmsprop',loss='mse')
        
    def store_transition(self,_state,a,r,state_):
        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0
        index = self.memory_counter%self.memory_size
        self.memory[index,:] = np.hstack([_state,a,r,state_])
        self.memory_counter+=1
        
    def choose_action(self,_state):
        _state = _state[np.newaxis,:]#2D array should be input to the net
        if np.random.rand()>self.epsilon: 
            q_eval = self.model1.predict(_state)
            action = np.argmax(q_eval)
        else:
            action = np.random.randint(0,4)
            
        return action
    
    def target_replace_op(self):
        v1 = self.model1.get_weights()
        self.model2.set_weights(v1)
        print('params have changed')
        
    def learn(self):
        if self.learn_step_count%self.replace_iter==0:
            self.target_replace_op()
#        if self.learn_step_count%(self.replace_iter*2)==0:
#            self.model1.save('model1.h5')
#            print('model saved done')
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index,:]
        q_eval,q_next = self.model1.predict(
                batch_memory[:,:self.n_features]),self.model2.predict(
                batch_memory[:,-self.n_features:])
        q_target = q_eval.copy()
        action = batch_memory[:,self.n_features].astype(int)
        batch_index = np.arange(self.batch_size,dtype=np.int32)
        reward = batch_memory[:,self.n_features+1]
        q_target[batch_index,action]=reward+self.gamma*np.max(q_next,1)
        self.history = self.model1.fit(batch_memory[:,:self.n_features],q_target,verbose=0,epochs=1)
        self.epsilon = self.epsilon*self.epsilon_decay if self.epsilon>0.01 else 0.01
        self.learn_step_count+=1
        self.all.append(self.history.history['loss'])
        
        
        
        
        
            
            

        
        
        
        
        
        
        
