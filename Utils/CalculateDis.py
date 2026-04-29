
import sys 
import numpy as np 


def calculate_dis_min(point1, locations):
    if len(locations) == 0 :
        return 0 
    
    dis_min = sys.maxsize 
    for i in range(len(locations)):
        point2 = locations[i]
        
        euclidean_dis = np.sqrt(np.sum((point1 - point2)**2)) 
        dis_min = min(euclidean_dis, dis_min)
        
    return dis_min



def calculate_dis_max(point1, locations):
    if len(locations) == 0 :
        return 50
    
    dis_max = -999 
    for i in range(len(locations)):
        point2 = locations[i]
        
        euclidean_dis = np.sqrt(np.sum((point1 - point2)**2)) 
        dis_max = max(euclidean_dis, dis_max)
        
    return dis_max