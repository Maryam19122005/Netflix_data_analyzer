# Netflix_data_analyzer
# Netflix Data Analysis - ES111 Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ======================
# 1. Data Preparation
# ======================
# Load dataset (replace with your file path)
df = pd.read_csv('Netflix_analysis.csv')

# Data cleaning
df['duration_min'] = df.apply(lambda x: int(x['duration'].split()[0]) 
                         if 'min' in str(x['duration']) else np.nan, axis=1)
df['seasons'] = df.apply(lambda x: int(x['duration'].split()[0]) 
                     if 'Season' in str(x['duration']) else np.nan, axis=1)

# ======================
# 2. Descriptive Statistics
# ======================
print("\n=== Basic Statistics ===")
print("Average movie duration:", df[df['type']=='Movie']['duration_min'].mean())
print("Average TV show seasons:", df[df['type']=='TV Show']['seasons'].mean())
print("Variance of release years:", df['release_year'].var())

# ======================
# 3. Visualizations
# ======================
# Histogram - Release Years
plt.figure(figsize=(10,6))
plt.hist(df['release_year'], bins=30, color='#E50914', edgecolor='black')
plt.title('Netflix Content by Release Year', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Titles', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.savefig('release_year_hist.png', dpi=300, bbox_inches='tight')
plt.show()

# Pie Chart - Content Types
plt.figure(figsize=(8,8))
df['type'].value_counts().plot.pie(colors=['#E50914','#221F1F'], 
                                 autopct='%1.1f%%',
                                 startangle=90,
                                 textprops={'fontsize': 12})
plt.title('Content Type Distribution', fontsize=14)
plt.ylabel('')
plt.savefig('content_type_pie.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================
# 4. Interval Estimation
# ======================
# 80/20 data split
train = df.sample(frac=0.8, random_state=42)
test = df.drop(train.index)

# 95% Confidence Interval for Release Year
ci = stats.norm.interval(0.95, 
                        loc=train['release_year'].mean(), 
                        scale=train['release_year'].sem())
print("\n=== Interval Estimation ===")
print(f"95% CI for Release Year: ({ci[0]:.1f}, {ci[1]:.1f})")

# Tolerance Interval (using 2σ ~ 95%)
lower = train['release_year'].mean() - 2*train['release_year'].std()
upper = train['release_year'].mean() + 2*train['release_year'].std()
coverage = ((test['release_year'] >= lower) & (test['release_year'] <= upper)).mean()
print(f"Tolerance Interval Coverage: {coverage*100:.1f}% of test data")

# ======================
# 5. Hypothesis Testing
# ======================
# Prepare duration data
movies = df[df['type']=='Movie']['duration_min'].dropna()
tv_shows = df[df['type']=='TV Show']['seasons'].dropna()

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(movies, tv_shows, equal_var=False)
print("\n=== Hypothesis Test ===")
print(f"T-statistic: {t_stat:.1f}, p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Conclusion: Significant difference in duration (movies vs TV shows)")
else:
    print("Conclusion: No significant difference")

# ======================
# 6. Frequency Distribution
# ======================
print("\n=== Frequency Distribution ===")
print("Top 5 countries:")
print(df['country'].str.split(', ').explode().value_counts().head(5))

# Save cleaned data
df.to_csv('cleaned_netflix_data.csv', index=False)
