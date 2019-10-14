# -*- coding: utf-8 -*-
import numpy as np

class snake():
     
    def __init__(self,width,length):
        self.width = width
        self.length = length
        self.room,self.body,self.food,self.state = self.state_initial()
        self.game_over=False
    def state_initial(self):      
        
        state = np.zeros(shape=[self.width,self.length],dtype=int)
        state[1,0]=1;state[1,1]=1
        body = np.vstack(np.where(state==1))
        room = np.vstack(np.where(state==0))
        p = np.random.randint(np.shape(room)[1])
        food = room[:,p]
        state[food[0],food[1]]=-1
        room = np.delete(room,p,1)
        return room,body,food,state
    
    def move(self,action):
        _state = self.state.copy()
        _body = self.body.copy()
        _length = len(self.body)
        action_space=['up','down','left','right']
        self.action = action_space[action]
        if self.action == 'up':
            ahead = self.body[-1]+[-1,0]
        elif self.action == 'down':
            ahead = self.body[-1]+[1,0]
        elif self.action == 'left':
            ahead = self.body[-1]+[0,-1]
        else :
            ahead = self.body[-1]+[0,1]            
        
        if self.judge(ahead)==False:#判断下一步有没有撞到自己
            self.game_over = True
            reward = -0.5
#            print('You have lost the game')
        else:
            self.body = np.vstack((self.body,ahead))#蛇的身体边长
            self.state[ahead[0],ahead[1]] = 1
            if (ahead!=self.food).any():#没吃到食物，删除蛇的尾巴
                if (ahead==self.body[0,:]).all():#如果首尾相连
                    pass
                else:
                    self.state[self.body[0][0],self.body[0][1]] = 0
                self.body = np.delete(self.body,0,0)  
            else:
                if np.sum(self.state)!=self.length*self.width:#如果身体填满整个空间就不去产生食物
                    self.room = np.vstack(np.where(self.state==0))
                    p = np.random.randint(np.shape(self.room)[1])
                    self.food = self.room[:,p]
                    self.state[self.food[0],self.food[1]]=-1
                
            if len(self.body)>_length:
                reward = 0.5
            else:
                reward = 0
            if len(self.body)==self.length*self.width:
                self.game_over = True
                reward = 10
                print('Winner!!!')
        _state = np.hstack((np.reshape(_state,-1),_body[-1])) 
        state_ = np.hstack((np.reshape(self.state,-1),self.body[-1]))
#        print('your action is :{} your reward is:{}'.format(action,reward))
        return _state,action,reward,state_
                
    def judge(self,ahead):
        judge = True
        if ahead[0]>=self.width or ahead[0]<0: 
            judge =  False
        if ahead[1]>=self.length or ahead[1]<0:
            judge =  False 
        if len(self.body[:,0])==2 and (ahead==self.body[0,:]).all():
            judge=False
        if (np.sum(ahead==self.body[1:,:],1)>=2).any():
            judge = False
        
                     
        return judge
            
if __name__=='__main__':
    G = snake(3,3)
    for i in range(1000):
        print(G.state)
        print('please choice a action:')
        action = int(input())
        G.move(action)            
        
        

            
        
            
            
        
            
        
        
        
        
        
        
        
    