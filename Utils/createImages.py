import matplotlib.pyplot as plt 
import os 
import numpy as np 

def smoothed_curve(data, level = 20):
    n  = len(data)
    smoothed_data = []
    for i in range(n-level):
        smoothed_data.append(np.mean(data[i:i+level]))
        
    return smoothed_data

def create_images(train_info):
    if not os.path.isdir("images"):
        os.mkdir("images")
        
    plt.figure(1)
    plt.plot(train_info.average_accumulated_reward_vec)
    plt.plot(smoothed_curve(data = train_info.average_accumulated_reward_vec, level = 20))
    plt.ylabel("average_accumulated_reward_vec")
    plt.xlabel("#")
    plt.savefig('images/average_accumulated_reward_vec.png')
    plt.close()

    plt.figure(2)
    plt.plot(train_info.cq_mean_vec_train)
    plt.plot(smoothed_curve(data = train_info.cq_mean_vec_train, level = 20))
    plt.ylabel("cq_mean_vec_train")
    plt.xlabel("#")
    plt.savefig('images/cq_mean_vec_train.png')
    plt.close()

    plt.figure(3)
    plt.plot(train_info.number_of_users_above_90)
    plt.plot(smoothed_curve(data = train_info.number_of_users_above_90, level = 20))
    plt.ylabel("number_of_users_above_90")
    plt.xlabel("#")
    plt.savefig('images/number_of_users_above_90.png')
    plt.close()

    plt.figure(4)
    plt.plot(train_info.ws_vec_train)
    plt.plot(smoothed_curve(data = train_info.ws_vec_train, level = 20))
    plt.ylabel("weighted_score_vec_train")
    plt.xlabel("#")
    plt.savefig('images/weighted_score_vec_train.png')
    plt.close()

    plt.figure(5)
    plt.plot(train_info.ct_score_vec_train)
    plt.plot(smoothed_curve(data = train_info.ct_score_vec_train, level = 20))
    plt.ylabel("ct_score_vec_train")
    plt.xlabel("#")
    plt.savefig('images/ct_score_vec_train.png')
    plt.close()

    plt.figure(6)
    plt.plot(train_info.cq_min)
    plt.plot(smoothed_curve(data = train_info.cq_min, level = 20))
    plt.ylabel("cq_min")
    plt.xlabel("#")
    plt.savefig('images/cq_min.png')
    plt.close()

