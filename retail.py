import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("retail_sales.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# -------------------------------
# Date Features
# -------------------------------
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month

# -------------------------------
# Visualizations
# -------------------------------

# Sales Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Total_Amount"], bins=10, kde=True)
plt.title("Sales Distribution")
plt.show()

# Monthly Sales
monthly = df.groupby("Month")["Total_Amount"].sum()

plt.figure(figsize=(7,5))
monthly.plot(kind="bar")
plt.title("Monthly Sales")
plt.ylabel("Total Sales")
plt.show()

# Category Sales
plt.figure(figsize=(7,5))
sns.barplot(data=df,
            x="Product_Category",
            y="Total_Amount",
            estimator=sum)

plt.title("Sales by Category")
plt.show()

# Correlation
plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include='number').corr(),
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# Encoding
# -------------------------------
le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["Product_Category"] = le.fit_transform(df["Product_Category"])

# -------------------------------
# Features & Target
# -------------------------------
X = df[["Gender",
        "Age",
        "Product_Category",
        "Quantity",
        "Price_per_Unit",
        "Month"]]

y = df["Total_Amount"]

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Model
# -------------------------------
model = LinearRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("\nPredicted Values")
print(prediction)

print("\nMAE:", mean_absolute_error(y_test, prediction))
print("R2 Score:", r2_score(y_test, prediction))

# -------------------------------
# Actual vs Predicted
# -------------------------------
result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": prediction
})

print("\nComparison")
print(result)

plt.figure(figsize=(8,5))
plt.scatter(y_test, prediction)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted")
plt.show()