
import pickle 
import matplotlib.pyplot as plt 
import numpy as np 

plt.close('all')

with open("train_info.pk","rb") as f:
    train_info = pickle.load(f)
    
    
plt.figure(1)
plt.plot(train_info.average_accumulated_reward_vec)


kernel_size = 10
kernel = np.ones(kernel_size) / kernel_size

smooted_r_vec =  np.convolve(train_info.average_accumulated_reward_vec, kernel, mode='same')
    
plt.plot(smooted_r_vec[:-kernel_size])


plt.figure(2)
data = train_info.cq_mean_vec_train
plt.plot(data)


kernel_size = 10
kernel = np.ones(kernel_size) / kernel_size

smooted_cq=  np.convolve(data, kernel, mode='same')
    
plt.plot(smooted_cq[:-kernel_size])


plt.figure(3)
data = train_info.average_changed_channels_vec
plt.plot(data)


kernel_size = 10
kernel = np.ones(kernel_size) / kernel_size

smooted_ancc=  np.convolve(data, kernel, mode='same')
    
plt.plot(smooted_ancc[:-kernel_size])