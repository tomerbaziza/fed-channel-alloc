
import numpy as np 

def watts_to_dbm(power_watts):
    if power_watts == 0:
        return -99999
    
    # print("Power in watts:", power_watts)
    return 10*np.log10(power_watts * 1000)


def dbm_to_watts(power_dbm):
    return 10**(power_dbm / 10) / 1000

def db_to_watts(x):
    return 10**(x/10) # Watts