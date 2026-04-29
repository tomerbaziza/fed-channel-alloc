
from scipy.spatial.distance import pdist
import math 


SPECTRUM = [208 + 2*i for i in range(10)] #[368 + 2*i for i in range(30)]#MHz


def get_path_loss(user1, user2, channel, channel_model = 'Egli'):
    # Compute channel path loss between two users in a specific channel.
    # the formula for Egli channel path loss is:
    #       LP = 40log10(d) - 20log10(40/f_c) - 20log10(H_T*H_R) - 10log10((G_T*G_R))
    # where d is distance in m, and f_c is channel frequency in MHz, h and g are antennas height and gain.
    h_t, h_r, g_t, g_r = user1.H, user2.H, user1.G, user2.G
    f_c = SPECTRUM[channel] #368 + channel*2      #We operate on the 100Mhz, and the specific channel adds a bit more
    distance = pdist((user1.location, user2.location))
    if channel_model == 'Egli':
        x =  40*math.log10(distance) - 20*math.log10(40/f_c) - 20*math.log10(h_t*h_r) - 10*math.log10(g_t*g_r) # [dB]
        return x #dB
    
    
    


# def get_attenuation(target_channel, inter_channel): # this is corresponds to dBc
#     # the loss in dBm with respect to the spectral distance between the channels
#     global SPECTRUM
#     spec = SPECTRUM
#     spectral_distance = abs(target_channel-inter_channel)
#     if spectral_distance == 0:
#         return 0
#     elif spectral_distance == 1:
#         return 24.5/2 #50
#     elif spectral_distance == 2:
#         return 42/2 #70
#     elif spectral_distance == 3:
#         return 55/2 #80
#     elif spectral_distance == 4:
#         return 61/2 #90
#     elif abs(spec[inter_channel] - spec[target_channel]) / spec[target_channel] <= 0.05 :
#         return 90/2
    
#     else:
#         return 110
    
def get_attenuation(target_channel, inter_channel): # this is corresponds to dBc
    # the loss in dBm with respect to the spectral distance between the channels
    global SPECTRUM
    spec = SPECTRUM
    spectral_distance = abs(target_channel-inter_channel)
    if spectral_distance == 0:
        return 0
    elif spectral_distance == 1:
        return 20 # 24.5 #50
    elif spectral_distance == 2:
        return 40 #42 #70
    elif spectral_distance == 3:
        return 50 #55 #80
    elif spectral_distance == 4:
        return 60 #61 #90
    
    elif abs(spec[inter_channel] - spec[target_channel]) / spec[target_channel] <= 0.05 :
        return 90
    
    else:
        return 110

    ## BASED ON Amos --> Skirt atttenuation 5% -90 
    ## BASED ON Amos --> Skirt atttenuation 10% -110 
# def get_attenuation_simulationBased(target_channel, inter_channel): # this is corresponds to dBc
#     # the loss in dBm with respect to the spectral distance between the channels
#     global SPECTRUM
#     spec = SPECTRUM
#     ch1_MHz  = spec[target_channel]
#     ch2_MHz = spec[inter_channel]
    
#     #### 1M , 2, 3, 4 --> 30, 42, 50, 60
#     spectral_distance = abs(ch1_MHz - ch2_MHz)
    
#     distance = spectral_distance // 1 # <- to acheive the interval

#     if distance <=1:
#         y1 = 0 
#         y2 = 30
        
#         x1 = 0
#         x2 = 1
        
#     elif distance >1 and distance <= 2:
#         x1 = 1 
#         y1 = 30
        
#         x2 = 2
#         y2 = 42
        
#     elif distance > 2 and distance <=3:
#         x1 = 2 
#         y1 = 42
        
#         x2 = 3
#         y2 = 50
        
#     else:
        
#         x1 = 3
#         y1 = 50
        
#         x2 = 4
#         y2 = 60
        
    
#     ## Interpolarion 
#     x3 = distance
#     y3 = (y2 - y1)*(x3 - x1)/(x2 - x1) + y1
#     return y3