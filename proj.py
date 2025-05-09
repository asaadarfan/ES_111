import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

# Load dataset
data = pd.read_csv("train.csv")

# Basic data cleaning
data = data.dropna(subset=['Age', 'Fare', 'Sex'])  # Ensure no missing values in key columns used for plots
data['Sex'] = data['Sex'].str.lower().str.strip()  # Normalize gender text

age_data = data['Age']

# Overall stats
mean_age = np.mean(age_data)
var_age = np.var(age_data)
print(f"Mean Age (Overall): {mean_age}")
print(f"Variance in Age (Overall): {var_age}")

# Plot histogram for overall age distribution
plt.hist(age_data, bins=20, edgecolor='black', alpha=0.6, label='All Passengers')

# Survivors' data
survived_age_data = data[data['Survived'] == 1]['Age']
mean_age_survived = np.mean(survived_age_data)
var_age_survived = np.var(survived_age_data)
print(f"Mean Age (Survivors): {mean_age_survived}")
print(f"Variance in Age (Survivors): {var_age_survived}")

# Plot histogram for survivors' age
plt.hist(survived_age_data, bins=20, edgecolor='black', alpha=0.6, label='Survivors')
plt.title('Age Distribution: All vs Survivors')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Age group pie charts
bins = [0, 18, 30, 45, 60, 100]
labels = ['0-17', '18-29', '30-44', '45-59', '60+']
age_groups = pd.cut(age_data, bins=bins, labels=labels)
age_groups_survived = pd.cut(survived_age_data, bins=bins, labels=labels)

group_counts = age_groups.value_counts().sort_index()
group_counts_survived = age_groups_survived.value_counts().sort_index()

# Pie chart for all passengers
group_counts.plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Age Distribution (All Passengers)')
plt.ylabel('')
plt.show()

# Pie chart for survivors only
group_counts_survived.plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Age Distribution (Survivors Only)')
plt.ylabel('')
plt.show()

# Mean comparison bar plot
plt.bar(['All Passengers', 'Survivors'], [mean_age, mean_age_survived], color=['skyblue', 'lightgreen'])
plt.title('Mean Age Comparison')
plt.ylabel('Mean Age')
plt.show()

# NEW: Bar graph of survivors by gender
survivors = data[data['Survived'] == 1]
gender_counts = survivors['Sex'].value_counts()

plt.bar(gender_counts.index, gender_counts.values, color=['orange', 'teal'])
plt.title('Survivors by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Survivors')
plt.show()

# Scatter plot between Age and Fare
plt.scatter(data['Age'], data['Fare'], alpha=0.6, edgecolors='w')
plt.title('Scatter Plot: Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.grid(True)
plt.show()

# Frequency distribution and weighted stats
frequency_distribution = age_data.value_counts().sort_index()
total_count = frequency_distribution.sum()

weighted_mean = np.sum(frequency_distribution.index * frequency_distribution) / total_count
squared_diff = (frequency_distribution.index - weighted_mean) ** 2
weighted_variance = np.sum(squared_diff * frequency_distribution) / total_count

print(f"Weighted Mean: {weighted_mean}")
print(f"Weighted Variance: {weighted_variance}")

# Train-validation split
train_data = age_data.sample(frac=0.8, random_state=42)
test_data = age_data.drop(train_data.index)

# Confidence Interval for Mean
confidence_interval = stats.t.interval(0.95, len(train_data)-1, loc=np.mean(train_data), scale=stats.sem(train_data))
print(f"Confidence Interval for the Mean: {confidence_interval}")

# Confidence Interval for Variance
n = len(train_data)
sample_variance = np.var(train_data, ddof=1)
chi2_lower = stats.chi2.ppf(0.025, df=n-1)
chi2_upper = stats.chi2.ppf(0.975, df=n-1)
ci_variance = ((n-1)*sample_variance / chi2_upper, (n-1)*sample_variance / chi2_lower)
print(f"Confidence Interval for the Variance: {ci_variance}")

# 95% Tolerance Interval
train_mean = np.mean(train_data)
train_std = np.std(train_data, ddof=1)
k = stats.t.ppf(0.975, df=n-1) * np.sqrt(1 + 1/n)
tolerance_interval = (train_mean - k*train_std, train_mean + k*train_std)
print(f"95% Tolerance Interval: {tolerance_interval}")

# Validation using test set
within_interval = test_data.between(tolerance_interval[0], tolerance_interval[1])
proportion_within = within_interval.mean()
print(f"Proportion of test data within tolerance interval: {proportion_within*100:.2f}%")

# Hypothesis test: Mean age = 30?
train_data_mean = np.mean(train_data)
train_data_std = np.std(train_data, ddof=1)
std_error = train_data_std / np.sqrt(n)

hyp_mean = 30
t_statistic = (train_data_mean - hyp_mean) / std_error
t_actual = stats.t.ppf(0.975, df=n-1)

print(f"t-critical: {t_actual}")    
print(f"t-statistic: {t_statistic}")

if -t_actual < t_statistic < t_actual:
    print("Fail to reject the null hypothesis (mean might be 30)")
else:
    print("Reject the null hypothesis (mean likely not 30)")