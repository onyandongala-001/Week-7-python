# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display the first few rows
print("First 5 rows of the dataset:")
print(iris_df.head())
print("\n" + "="*50 + "\n")

# Explore the structure of the dataset
print("Dataset information:")
print(iris_df.info())
print("\n" + "="*50 + "\n")

# Check for missing values
print("Missing values in each column:")
print(iris_df.isnull().sum())
print("\n" + "="*50 + "\n")

# Since there are no missing values, no cleaning is needed
print("No missing values found. Dataset is clean.")
print("\n" + "="*50 + "\n")

# Compute basic statistics for numerical columns
print("Basic statistics of numerical columns:")
print(iris_df.describe())
print("\n" + "="*50 + "\n")

# Group by species and compute mean of numerical columns
print("Mean values by species:")
species_group = iris_df.groupby('species').mean()
print(species_group)
print("\n" + "="*50 + "\n")

# Identify patterns and interesting findings
print("Interesting findings:")
print("1. Setosa has significantly smaller petal dimensions compared to other species.")
print("2. Virginica has the largest sepal length on average.")
print("3. Versicolor has medium-sized measurements across all features.")
print("\n" + "="*50 + "\n")

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Analysis and Visualization', fontsize=16, fontweight='bold')

# 1. Line chart (simulated time series by index)
axes[0, 0].plot(iris_df.index, iris_df['sepal length (cm)'], label='Sepal Length', color='blue')
axes[0, 0].plot(iris_df.index, iris_df['petal length (cm)'], label='Petal Length', color='red')
axes[0, 0].set_title('Trend of Sepal and Petal Length by Index')
axes[0, 0].set_xlabel('Index')
axes[0, 0].set_ylabel('Length (cm)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Bar chart (average sepal length by species)
species_means = iris_df.groupby('species').mean()
species_means['sepal length (cm)'].plot(kind='bar', ax=axes[0, 1], color=['lightblue', 'lightgreen', 'lightcoral'])
axes[0, 1].set_title('Average Sepal Length by Species')
axes[0, 1].set_xlabel('Species')
axes[0, 1].set_ylabel('Sepal Length (cm)')
axes[0, 1].tick_params(axis='x', rotation=0)

# 3. Histogram (distribution of petal length)
axes[1, 0].hist(iris_df['petal length (cm)'], bins=15, color='purple', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Distribution of Petal Length')
axes[1, 0].set_xlabel('Petal Length (cm)')
axes[1, 0].set_ylabel('Frequency')

# 4. Scatter plot (sepal length vs petal length, colored by species)
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for species, color in colors.items():
    species_data = iris_df[iris_df['species'] == species]
    axes[1, 1].scatter(species_data['sepal length (cm)'], species_data['petal length (cm)'], 
                       label=species, color=color, alpha=0.7)
axes[1, 1].set_title('Sepal Length vs Petal Length by Species')
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Create a pairplot to show relationships between all variables
print("Pairplot of all variables (this may take a moment)...")
sns.pairplot(iris_df, hue='species', palette=colors)
plt.suptitle('Pairplot of Iris Dataset by Species', y=1.02)
plt.show()

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
numeric_df = iris_df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Create a boxplot to show distribution of features by species
plt.figure(figsize=(12, 8))
iris_df_melted = pd.melt(iris_df, id_vars="species", var_name="features", value_name="value")
plt.figure(figsize=(10, 8))
sns.boxplot(x="features", y="value", hue="species", data=iris_df_melted, palette=colors)
plt.title('Distribution of Features by Species')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print a summary of findings
print("\n" + "="*50)
print("SUMMARY OF FINDINGS")
print("="*50)
print("1. The dataset contains 150 samples with 4 numerical features and 1 categorical feature (species).")
print("2. There are no missing values in the dataset.")
print("3. Setosa has the smallest petals and sepals on average.")
print("4. Virginica has the largest petals and sepals on average.")
print("5. There is a strong positive correlation between petal length and petal width (0.96).")
print("6. There is a moderate positive correlation between sepal length and petal length (0.87).")
print("7. The distribution of petal length is bimodal, reflecting the different species.")
print("8. Setosa is clearly distinguishable from the other two species based on petal measurements.")
print("9. Versicolor and Virginica have some overlap in their measurements, making them harder to distinguish.")