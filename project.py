import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


data = pd.read_csv("train.csv")
age_data = data['Age'].dropna()
mean_age = np.mean(age_data)
var_age = np.var(age_data)

print(f"mean age is: {mean_age}")
print(f"variance in age is: {var_age}")

plt.hist(age_data, bins=20, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

bins = [0, 18, 30, 45, 60, 100]
labels = ['0-17', '18-29', '30-44', '45-59', '60+']
age_groups = pd.cut(age_data, bins=bins, labels=labels)

# Count how many passengers fall into each age group
group_counts = age_groups.value_counts().sort_index()

# Plot pie chart
group_counts.plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Passenger Age Distribution')
plt.ylabel('')  
plt.show()

frequency_distribution = age_data.value_counts().sort_index()
print(frequency_distribution)
total_count = frequency_distribution.sum()

# mean calculation  
weighted_mean = np.sum(frequency_distribution.index * frequency_distribution) / total_count

# Variance calculation
squared_diff = (frequency_distribution.index - weighted_mean) ** 2
weighted_variance = np.sum(squared_diff * frequency_distribution) / total_count

print(f"Weighted Mean: {weighted_mean}")
print(f"Weighted Variance: {weighted_variance}")


# Split dataset: 80% for training, 20% for validation
train_data = age_data.sample(frac=0.8, random_state=42)
test_data = age_data.drop(train_data.index)

# 95% confidence interval for the mean
confidence_interval = stats.t.interval(0.95, len(train_data)-1, loc=np.mean(train_data), scale=stats.sem(train_data))

print(f"Confidence Interval for the Mean: {confidence_interval}")

# Hypothesis test: Testing if mean age is 30 
n = len(train_data)
train_data_mean = np.mean(train_data)
train_data_std = np.std(train_data, ddof=1)
std_error = train_data_std / np.sqrt(n)

hyp_mean = 30
t_statistic = (train_data_mean-hyp_mean)/std_error

t_actual = stats.t.ppf(0.975, df = n-1)
print(t_actual)
print(t_statistic)    

if -t_actual < t_statistic < t_actual:
    print("Fail to reject the null hypothesis (mean might be 30)")
else:
    print("Reject the null hypothesis (mean likely not 30)")
