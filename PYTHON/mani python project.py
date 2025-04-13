import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
data = pd.read_csv("C:\\Users\\Mahendra\\Downloads\\Mental_Health_Care_in_the_Last_4_Weeks.csv")

# Clean column names and types
data.columns = data.columns.str.strip()
data.dropna(subset=['Value'], inplace=True)
data['Time Period Start Date'] = pd.to_datetime(data['Time Period Start Date'])
data['Time Period End Date'] = pd.to_datetime(data['Time Period End Date'])

# -------------------------------
# BASIC INFO
# -------------------------------
print("\n=== BASIC INFO ===")
print(data.info())
print("\nSummary Stats:")
print(data['Value'].describe())

# -------------------------------
# DISTRIBUTION PLOT
# -------------------------------
plt.figure()
sns.histplot(data['Value'], bins=30, kde=True, color='mediumseagreen')
plt.title("Distribution of Mental Health Care Access in Last 4 Weeks")
plt.xlabel("Value (%)")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# TREND OVER TIME
# -------------------------------
trend = data.groupby('Time Period Label')['Value'].mean().sort_index()
plt.figure()
sns.lineplot(x=trend.index, y=trend.values, marker='o', linewidth=2)
plt.title("National Average Mental Health Care Access Over Time")
plt.xlabel("Time Period")
plt.ylabel("Average Value (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# TOP STATES (Most Recent Time Period)
# -------------------------------
latest_period = data['Time Period'].max()
latest_data = data[data['Time Period'] == latest_period]
state_avg = latest_data.groupby('State')['Value'].mean().sort_values(ascending=False).head(10)

plt.figure()
sns.barplot(x=state_avg.values, y=state_avg.index)
plt.title("Top 10 States by Mental Health Care Access (Most Recent Period)")
plt.xlabel("Value (%)")
plt.ylabel("State")
plt.show()

# -------------------------------
# BOXPLOT BY SUBGROUP
# -------------------------------
plt.figure()
sns.boxplot(x='Subgroup', y='Value', data=data)
plt.title("Mental Health Care Access by Subgroup")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# CONFIDENCE INTERVALS (MOST RECENT)
# -------------------------------
ci_data = latest_data.groupby('State')[['Value', 'LowCI', 'HighCI']].mean().dropna().sort_values('Value', ascending=False).head(10)
plt.figure()
plt.errorbar(ci_data.index, ci_data['Value'], 
             yerr=[ci_data['Value'] - ci_data['LowCI'], ci_data['HighCI'] - ci_data['Value']],
             fmt='o', capsize=5, color='darkblue')
plt.xticks(rotation=45)
plt.title("Top 10 States with Confidence Intervals")
plt.ylabel("Value (%)")
plt.tight_layout()
plt.show()

# -------------------------------
# Z-TEST FUNCTION
# -------------------------------
def perform_z_test(group1, group2, label1, label2):
    """
    Perform Z-test comparing two subgroups or states based on mean Value.
    """
    x1 = group1['Value'].mean()
    s1 = group1['Value'].std()
    n1 = len(group1)

    x2 = group2['Value'].mean()
    s2 = group2['Value'].std()
    n2 = len(group2)

    # Standard error
    se = np.sqrt((s1**2 / n1) + (s2**2 / n2))
    z = (x1 - x2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    print(f"\n===== Z-TEST: {label1} vs {label2} =====")
    print(f"Mean {label1}: {x1:.2f}%, n = {n1}")
    print(f"Mean {label2}: {x2:.2f}%, n = {n2}")
    print(f"Z-Score: {z:.4f}")
    print(f"P-Value: {p_value:.4f}")
    if p_value < 0.05:
        print("Conclusion: Significant difference (p < 0.05)")
    else:
        print("Conclusion: No significant difference (p ≥ 0.05)")

# -------------------------------
# Z-TEST: Example - Comparing Two States
# -------------------------------
state1 = 'California'
state2 = 'Texas'
group1 = data[data['State'] == state1]
group2 = data[data['State'] == state2]
perform_z_test(group1, group2, state1, state2)


# -------------------------------
# HEATMAP: CORRELATION BETWEEN NUMERIC FIELDS
# -------------------------------
numeric_cols = ['Value', 'LowCI', 'HighCI']
corr_matrix = data[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Mental Health Metrics")
plt.tight_layout()
plt.show()

# -------------------------------
# PAIRPLOT: NUMERIC VARIABLES
# -------------------------------
sns.pairplot(data[numeric_cols].dropna())
plt.suptitle("Pairwise Plots of Mental Health Metrics", y=1.02)
plt.show()

# -------------------------------
# HISTPLOTS FOR EACH NUMERIC COLUMN
# -------------------------------
for col in numeric_cols:
    plt.figure()
    sns.histplot(data[col].dropna(), bins=30, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


