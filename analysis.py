# ============================================================
#  Airbnb NYC Price Analysis & Predictor
#  Author  : Your Name
#  Dataset : airbnb.csv (included in this repo)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# ---------- Load Data ----------
df = pd.read_csv('airbnb.csv')
print("=" * 50)
print("AIRBNB NYC PRICE ANALYSIS")
print("=" * 50)
print(f"\nListings: {len(df)}")
print(df.head())
print("\nStats:")
print(df[['price','minimum_nights','number_of_reviews','availability_365']].describe().round(2))

# Remove extreme outliers (top 1% prices)
df = df[df['price'] < df['price'].quantile(0.99)]
df = df[df['price'] > 0]

# ---------- Plot 1: Price Distribution ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(df['price'], bins=40, color='#E74C3C', edgecolor='black', alpha=0.8)
axes[0].axvline(df['price'].mean(), color='navy', linestyle='--', linewidth=2,
                label=f"Mean: ${df['price'].mean():.0f}")
axes[0].axvline(df['price'].median(), color='green', linestyle='--', linewidth=2,
                label=f"Median: ${df['price'].median():.0f}")
axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Price per Night ($)')
axes[0].set_ylabel('Number of Listings')
axes[0].legend()

nb_price = df.groupby('neighbourhood')['price'].median().sort_values(ascending=False)
axes[1].bar(nb_price.index, nb_price.values,
            color=['#E74C3C','#3498DB','#2ECC71','#F39C12','#9B59B6'],
            edgecolor='black', width=0.6)
axes[1].set_title('Median Price by Neighbourhood', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Neighbourhood')
axes[1].set_ylabel('Median Price per Night ($)')
for i, v in enumerate(nb_price.values):
    axes[1].text(i, v+1, f'${v:.0f}', ha='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig('plot1_price_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot1_price_distribution.png")

# ---------- Plot 2: Room Type Analysis ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
rt_counts = df['room_type'].value_counts()
axes[0].pie(rt_counts, labels=rt_counts.index, autopct='%1.1f%%',
            colors=['#3498DB','#E74C3C','#2ECC71'], startangle=90, explode=[0.05]*3)
axes[0].set_title('Listings by Room Type', fontsize=14, fontweight='bold')

rt_price = df.groupby('room_type')['price'].median().sort_values(ascending=False)
axes[1].bar(rt_price.index, rt_price.values,
            color=['#3498DB','#E74C3C','#2ECC71'], edgecolor='black', width=0.5)
axes[1].set_title('Median Price by Room Type', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Room Type')
axes[1].set_ylabel('Median Price ($)')
for i, v in enumerate(rt_price.values):
    axes[1].text(i, v+1, f'${v:.0f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('plot2_room_type.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot2_room_type.png")

# ---------- Plot 3: Reviews vs Price & Availability ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(df['number_of_reviews'], df['price'], alpha=0.3,
                color='#9B59B6', s=20, edgecolors='none')
axes[0].set_title('Number of Reviews vs Price', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Reviews')
axes[0].set_ylabel('Price per Night ($)')
corr1 = df['number_of_reviews'].corr(df['price'])
axes[0].text(5, df['price'].max()*0.9, f'Corr: {corr1:.2f}', fontsize=11, color='darkred')

axes[1].scatter(df['availability_365'], df['price'], alpha=0.3,
                color='#E67E22', s=20, edgecolors='none')
axes[1].set_title('Availability (days/year) vs Price', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Availability (days per year)')
axes[1].set_ylabel('Price per Night ($)')
corr2 = df['availability_365'].corr(df['price'])
axes[1].text(5, df['price'].max()*0.9, f'Corr: {corr2:.2f}', fontsize=11, color='darkred')
plt.tight_layout()
plt.savefig('plot3_reviews_availability.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot3_reviews_availability.png")

# ---------- ML: Price Prediction with Random Forest ----------
print("\nTraining Random Forest price predictor...")
le_nb = LabelEncoder(); le_rt = LabelEncoder()
df['nb_enc'] = le_nb.fit_transform(df['neighbourhood'])
df['rt_enc'] = le_rt.fit_transform(df['room_type'])

features = ['nb_enc','rt_enc','minimum_nights','number_of_reviews',
            'reviews_per_month','availability_365','latitude','longitude']
X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"  MAE : ${mae:.2f}")
print(f"  R²  : {r2:.3f}")

# ---------- Plot 4: Feature Importance & Actual vs Predicted ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
importances = pd.Series(model.feature_importances_, index=features).sort_values()
importances.plot(kind='barh', ax=axes[0], color='#3498DB', edgecolor='black')
axes[0].set_title('Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Importance Score')

axes[1].scatter(y_test, y_pred, alpha=0.4, color='#E74C3C', s=20, edgecolors='none')
max_val = max(y_test.max(), y_pred.max())
axes[1].plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='Perfect fit')
axes[1].set_title(f'Actual vs Predicted Price\nR² = {r2:.3f}  |  MAE = ${mae:.0f}',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Actual Price ($)')
axes[1].set_ylabel('Predicted Price ($)')
axes[1].legend()
plt.tight_layout()
plt.savefig('plot4_model_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot4_model_results.png")

print("\n" + "=" * 50)
print("KEY FINDINGS")
print("=" * 50)
print(f"  Avg nightly price      : ${df['price'].mean():.0f}")
print(f"  Most expensive area    : {nb_price.idxmax()} (${nb_price.max():.0f})")
print(f"  Cheapest area          : {nb_price.idxmin()} (${nb_price.min():.0f})")
print(f"  Model R² score         : {r2:.3f}")
print(f"  Mean absolute error    : ${mae:.2f}")
print("\nAnalysis complete!")
