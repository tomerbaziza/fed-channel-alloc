import numpy as np 
import matplotlib.pyplot as plt 
import pickle  


def gaussian_distribution_1d(mean = 0, sigma = 1):
    
    def pdf(x):
        value = 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5 * ((x - mean)/sigma)**2)
        return value
    
    return pdf

def smooth_average(x, index=5):
    n  = len(x)
    cuumulated_sum = np.cumsum(x)
    part_1 = cuumulated_sum[index-1:]
    part_2 = np.concatenate(([0], cuumulated_sum[:-index]), axis = 0)
    
    smoothed = (part_1 - part_2) / index
    
    ## add the last points 
    # number of miising points == index 
    
    for i in range(n - index, n):
        val = np.mean(x[i - index - 1: i + 1])
        smoothed = np.concatenate((smoothed, [val]), axis = 0)
    
    return smoothed
    
    
plt.close('all')
#import matplotlib.pyplot as plt 

with open('data_with_bigger_than/dataaaStat.pk', 'rb') as file:
    statistical_data = pickle.load(file)

average_accumulated_reward_vec_all, average_changed_channels_vec_all = statistical_data
a1 = np.array(average_accumulated_reward_vec_all)
b1 = np.array(average_changed_channels_vec_all)
     
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.figure(1, figsize = (8,7))
number = [i +2 for i in range(23)]        
plt.boxplot(a1, labels = number)
plt.xlabel('Number of Nets')
plt.ylim(23,41)
plt.plot([min(number)-1,max(number)], [40,40], '--k')
plt.ylabel('Accumulated Reward')

plt.figure(2, figsize = (8,7))
number = [i +2 for i in range(23)]        
plt.boxplot(b1, labels = number)
plt.xlabel('Number of Nets')
plt.ylabel("Average number of channel changes")



with open('data_withoutBigerTHan/dataaaStat.pk', 'rb') as file:
    statistical_data = pickle.load(file)
    
average_accumulated_reward_vec_all, average_changed_channels_vec_all = statistical_data
a2 = np.array(average_accumulated_reward_vec_all)
b2 = np.array(average_changed_channels_vec_all)

plt.figure(3, figsize = (8,7))
number = [i +2 for i in range(23)]        
plt.boxplot(a2, labels = number)
plt.xlabel('Number of Nets')
plt.ylim(23,41)
plt.ylabel('Accumulated Reward')
plt.plot([min(number)-1,max(number)], [40,40], '--k')


plt.figure(4, figsize = (8,7))
number = [i +2 for i in range(23)]        
plt.boxplot(b2, labels = number)
plt.xlabel('Number of Nets')
plt.ylabel("Average number of channel changes")


## CReate an average plot !!!! and std avareg plot gauusians !! we 
# wish to achive high mean and low variance 
# reward
## 1== with >0 
## 2 == withput >0
vec_1_withBiggerThan = [a1, b1]
vec_2_withoutBiggerThan  = [a2, b2]
name = ['Accumulated Reward', 'Average number of channel changes']

for i,j, n in zip(vec_1_withBiggerThan, vec_2_withoutBiggerThan, name):
    
    i_mean  = np.mean(i, axis = 0)
    j_mean = np.mean(j, axis = 0)
    
    plt.figure(figsize = (8,7))
    plt.hist(i_mean, bins= 10, label = 'Median up')
    plt.hist(j_mean, bins = 10, label = 'No condition' )
    plt.xlabel(n)
    plt.legend(frameon = False)
    plt.ylabel('Frequency')
    
    plt.figure(figsize = (8,7))
    plt.plot(np.arange(len(i_mean))+2, i_mean, label = 'Median up')
    plt.plot(np.arange(len(j_mean))+2, j_mean, label = 'No condition')
    plt.ylabel(n)
    plt.xlabel('Number of Nets')
    plt.legend(frameon = False)


rewards_with_biggerThan_mean = np.mean(a1)
rewards_with_biggerThan_std = np.std(a1)

# average number of changes
noc_with_biggerThan_mean= np.mean(b1)
noc_with_biggerThan_std = np.std(b1)
#########################################
rewards_without_biggerThan_mean = np.mean(a2)
rewards_without_biggerThan_std = np.std(a2)

# average number of changes
noc_without_biggerThan_mean = np.mean(b2)
noc_without_biggerThan_std = np.std(b2)

print("rewards_without_biggerThan (mean, std):", (rewards_without_biggerThan_mean,
                                                  rewards_without_biggerThan_std))


print("rewards_with_biggerThan (mean, std):", (rewards_with_biggerThan_mean,
                                                  rewards_with_biggerThan_std))

print("Number of change channels_without_biggerThan (mean, std):", (noc_without_biggerThan_mean, 
                                                                    noc_without_biggerThan_std))

print("Number of change channels_with_biggerThan (mean, std):", (noc_with_biggerThan_mean, 
                                                                 noc_with_biggerThan_std))


print("Here you should insetrt a statistical test! To know which technique is better")

###########################################
# Statistical Test 
###########################################
from scipy.stats import norm 
vec_1_withBiggerThan = [a1, b1]
vec_2_withoutBiggerThan  = [a2, b2]
name = ['Accumulated Reward', 'Average number of channel changes']

a1_mean = np.mean(a1, axis = 0)
mean_a1_mean = np.mean(a1_mean)
se_a1 = np.std(a1_mean)


a2_mean = np.mean(a2, axis = 0)
mean_a2_mean = np.mean(a2_mean)
se_a2 = np.std(a2_mean)


delta_a = mean_a1_mean - mean_a2_mean
n  = len(a1_mean)
m = len(a2_mean)
std_delta  = np.sqrt((se_a1**2)/n + (se_a2 ** 2)/ m)

test_statistic = np.abs((delta_a - 0))/std_delta

p_val  = 2*(1 - norm.cdf(test_statistic))
print("P-value (accumulated reward):", p_val)

b1_mean = np.mean(b1, axis = 0)
mean_b1_mean = np.mean(b1_mean)
se_b1 = np.std(b1_mean)


b2_mean = np.mean(b2, axis = 0)
mean_b2_mean = np.mean(b2_mean)
se_b2 = np.std(b2_mean)


delta_b = mean_b1_mean - mean_b2_mean
n  = len(b1_mean)
m = len(b2_mean)
std_delta_b  = np.sqrt((se_b1**2)/n + (se_b2 ** 2)/ m)

test_statistic_b = abs((delta_b - 0))/std_delta_b
p_val_b  = 2*(1 - norm.cdf(test_statistic_b))
print("P-value (average numebr of change channel):", p_val_b)




###########################################
# The right way of doing the boostraping
###########################################

delta_boost_a = []
for i in range(50000):
    n1,c1 = np.shape(a1)
    
    a1_vec_smapled = []
    a2_vec_sampled = []
    for j in range(c1):
        a_1_column = np.random.choice(a1[:,j])
        a_2_column = np.random.choice(a2[:,j])
        
        a1_vec_smapled.append(a_1_column)
        a2_vec_sampled.append(a_2_column)
    
    mean_a1 = np.mean(a1_vec_smapled)
    mean_a2 = np.mean(a2_vec_sampled)
    delta = mean_a1 - mean_a2
    delta_boost_a.append(delta)
    
    
plt.figure(9000,figsize = (8,7))
hist_a1, bins_a1 = np.histogram(delta_boost_a, bins = 500)
delta_bins = bins_a1[1] - bins_a1[0]
area_under_the_curve = len(delta_boost_a) * delta_bins 


plt.stairs(values = hist_a1 , edges = bins_a1, fill = True ,color = 'k' )
plt.xlabel(r'$\delta$ Accumulated Rewards')
plt.ylabel("Frequency")


####  add Gausiaan distribution 
mean_dist = np.mean(delta_boost_a)
std = np.std(delta_boost_a)
pdf_a1 = gaussian_distribution_1d(mean = mean_dist, sigma = std)


vector_of_values = np.linspace(start = np.min(delta_boost_a), stop =  np.max(delta_boost_a), num = 10000)
pdf_values = [pdf_a1(i) * area_under_the_curve  for i in vector_of_values]
plt.plot(vector_of_values, pdf_values  , color = 'r')

## Now over the average number of change channels
delta_boost_b = []
for i in range(50000):
    n1,c1 = np.shape(b1)
    
    b1_vec_smapled = []
    b2_vec_sampled = []
    for j in range(c1):
        b_1_column = np.random.choice(b1[:,j])
        b_2_column = np.random.choice(b2[:,j])
        
        b1_vec_smapled.append(b_1_column)
        b2_vec_sampled.append(b_2_column)
    
    mean_b1 = np.mean(b1_vec_smapled)
    mean_b2 = np.mean(b2_vec_sampled)
    delta = mean_b1 - mean_b2
    delta_boost_b.append(delta)
       
plt.figure(9001, figsize = (8,7))
hist_b1, bins_b1 = np.histogram(delta_boost_b, bins = 500)
delta_bins = bins_b1[1] - bins_b1[0]
area_under_the_curve = len(delta_boost_b) * delta_bins 


plt.stairs(values = hist_b1  , edges = bins_b1, fill = True ,color = 'k' )
plt.xlabel(r'$\delta$ Average number of channel changes')
plt.ylabel("Frequency")


####  add Gausiaan distribution 
mean_dist = np.mean(delta_boost_b)
std = np.std(delta_boost_b)
pdf_b1 = gaussian_distribution_1d(mean = mean_dist, sigma = std)


vector_of_values = np.linspace(start = np.min(delta_boost_b), stop =  np.max(delta_boost_b), num = 10000)
pdf_values = [pdf_b1(i)  * area_under_the_curve for i in vector_of_values]
plt.plot(vector_of_values, pdf_values, color = 'r')


## Just for cheking 
area = 0
for i in range(len(pdf_values)-1):
    area += pdf_values[i] * (vector_of_values[i+1] - vector_of_values[i])
    
area2 = 0
for i in range(len(hist_b1)):
    area2 += hist_b1[i] *  (bins_b1[i+1] - bins_b1[i]) 
    
###########################################    
# Calculate P-value based bootstrapping
# P-vaue = P(Z > |Tn|) = P(Z<-Tn) + P(Z>Tn)
###########################################
Tn_a = test_statistic
Tn_b = test_statistic_b 



counter = 0
counter2 = 0
for val in delta_boost_a:
    if val < -Tn_a or val > Tn_a:
        counter += 1
    counter2 += 1
print("P-value for accumulated reward test (delta):", counter/counter2)


counter = 0
counter2 = 0
for val in delta_boost_b:
    if val < -Tn_b or val > Tn_b:
        counter += 1
    counter2 += 1
print("P-value for average number pf channel changes (delta):", counter/counter2)


###########################################
# Calculatign the confidence interval (95%, alpha 5%)
###########################################
# Normality: ##  numpy.quantile(values , [q1,q2,...])
    # q_alpha/2 = q_0.025-->1.96 (from Z plot)
normal_dist = norm(
    loc=0 , 
    scale= 1 
)
q_alpha_2 = normal_dist.ppf(0.975)

CI_normal_interval_a = [delta_a - std_delta*q_alpha_2,delta_a +std_delta*q_alpha_2]
CI_normal_interval_b = [delta_b - std_delta_b*q_alpha_2, delta_b +std_delta_b*q_alpha_2]
print("Confidence intervals:")
print("Normal Interval for Accumulated reward: ", CI_normal_interval_a)
print("Normal Interval for Average number of change channels:", CI_normal_interval_b)

percentile_interval_a = np.quantile(sorted(delta_boost_a), [0.025, 0.975])
percentile_interval_b = np.quantile(sorted(delta_boost_b), [0.025, 0.975])
###########################################
# Training performance comparison 
###########################################


# with open('data_with_bigger_than/dataaa.pk', 'rb') as file:
#     training_data_with_bigger_than = pickle.load(file)

# average_accumulated_reward_vec_all  , average_changed_channels_vec_all  = training_data_with_bigger_than

# a1 = np.array(average_accumulated_reward_vec_all)
# b1 = np.array(average_changed_channels_vec_all)
     

# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.size"] = 14


# fig, ax = plt.subplots( figsize = (7,6))
# a1_mean = np.mean(a1, axis = 0)

# ax.plot(a1_mean, color = 'r', alpha = 0.5)

# a1_std = np.std(a1, axis = 0)

# xx = np.arange(len(a1_std))
# ax.fill_between(xx,a1_mean - a1_std, [min(40, i) for i in a1_mean + a1_std], color = 'r', alpha  = 0.1 )

# ax.set_ylim(23,41)
# ax.plot([0,1600], [40,40], '--k')

# ax.set_xlim([0,1600])

# a1_smoothed = smooth_average(a1_mean, index  = 20)
# ax.plot(a1_smoothed, color = 'r')
# ax.set_ylabel('Accumulated Reward')
# ax.set_xlabel('#Episode')



# fig, ax = plt.subplots( figsize = (7,6))
# b1_mean = np.mean(b1, axis = 0)

# ax.plot(b1_mean, color = 'b', alpha = 0.5)

# b1_std = np.std(b1, axis = 0)

# xx = np.arange(len(b1_std))
# ax.fill_between(xx,[max(0,i) for i in b1_mean - b1_std], b1_mean + b1_std, color = 'b', alpha  = 0.1 )

# ax.set_ylim(0,None)
# ax.plot([0,1600], [1,1], '--k')

# ax.set_xlim([0,1600])

# b1_smoothed = smooth_average(b1_mean, index  = 20)
# ax.plot(b1_smoothed, color = 'b')
# plt.xlabel('#Episode')
# plt.ylabel("Average number of channel changes")

# plt.figure(10)
# plt.plot(a1_mean, color = 'r')

# ###########################################
# # Training performance comparison 
# ###########################################


# with open('data_withoutBigerTHan/dataaa.pk', 'rb') as file:
#     training_data_with_bigger_than = pickle.load(file)

# average_accumulated_reward_vec_all  , average_changed_channels_vec_all  = training_data_with_bigger_than

# a1 = np.array(average_accumulated_reward_vec_all)
# b1 = np.array(average_changed_channels_vec_all)
     

# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.size"] = 14


# fig, ax = plt.subplots( figsize = (7,6))
# a1_mean = np.mean(a1, axis = 0)

# ax.plot(a1_mean, color = 'r', alpha = 0.5)

# a1_std = np.std(a1, axis = 0)

# xx = np.arange(len(a1_std))
# ax.fill_between(xx,a1_mean - a1_std, [min(40, i) for i in a1_mean + a1_std], color = 'r', alpha  = 0.1 )

# ax.set_ylim(23,41)
# ax.plot([0,1600], [40,40], '--k')

# ax.set_xlim([0,1600])

# a1_smoothed = smooth_average(a1_mean, index  = 20)
# ax.plot(a1_smoothed, color = 'r')
# ax.set_ylabel('Accumulated Reward')
# ax.set_xlabel('#Episode')



# fig, ax = plt.subplots( figsize = (7,6))
# b1_mean = np.mean(b1, axis = 0)

# ax.plot(b1_mean, color = 'b', alpha = 0.5)

# b1_std = np.std(b1, axis = 0)

# xx = np.arange(len(b1_std))
# ax.fill_between(xx,[max(0,i) for i in b1_mean - b1_std], b1_mean + b1_std, color = 'b', alpha  = 0.1 )

# ax.set_ylim(0,None)
# ax.plot([0,1600], [1,1], '--k')

# ax.set_xlim([0,1600])

# b1_smoothed = smooth_average(b1_mean, index  = 20)
# ax.plot(b1_smoothed, color = 'b')
# plt.xlabel('#Episode')
# plt.ylabel("Average number of channel changes")



# plt.figure(10)
# plt.plot(a1_mean, color = 'g')
