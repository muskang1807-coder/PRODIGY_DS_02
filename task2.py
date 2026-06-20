import pandas as pd
import matplotlib.pyplot as plt

# Load Titanic Dataset
df = pd.read_csv(r"C:\Users\MUSKAN\Downloads\train.csv")

# Display basic information
print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# -----------------------------
# Visualization 1: Survival by Gender
# -----------------------------
survival_gender = df.groupby("Sex")["Survived"].sum()

plt.figure(figsize=(6,4))
survival_gender.plot(kind="bar")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Survivors")
plt.tight_layout()
plt.savefig("survival_by_gender.png")
plt.show()

# -----------------------------
# Visualization 2: Passenger Class Distribution
# -----------------------------
plt.figure(figsize=(6,4))
df["Pclass"].value_counts().sort_index().plot(kind="bar")
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("passenger_class.png")
plt.show()

# -----------------------------
# Visualization 3: Age Distribution
# -----------------------------
plt.figure(figsize=(8,5))
df["Age"].dropna().hist(bins=20)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.show()

# -----------------------------
# Insights
# -----------------------------
print("\nEDA COMPLETED SUCCESSFULLY!")
print("Generated Files:")
print("1. survival_by_gender.png")
print("2. passenger_class.png")
print("3. age_distribution.png")