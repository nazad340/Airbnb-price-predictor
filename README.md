# 🏠 Airbnb NYC Price Analysis & Predictor

EDA + Machine Learning model to predict Airbnb listing prices in New York City.

## 📊 What This Project Does
- Analyzes price distributions across NYC neighbourhoods
- Compares prices by room type and neighbourhood
- Trains a Random Forest model to predict nightly price
- Evaluates model performance with R² and MAE metrics

## 🗂️ Project Structure
```
5-airbnb-price-predictor/
├── analysis.py    ← Main Python script
├── airbnb.csv     ← Dataset (800 listings)
└── README.md
```

## 📈 Charts Generated
| File | Description |
|------|-------------|
| plot1_price_distribution.png | Price histogram + median by neighbourhood |
| plot2_room_type.png | Room type breakdown + price comparison |
| plot3_reviews_availability.png | Reviews & availability vs price |
| plot4_model_results.png | Feature importance + actual vs predicted |

## 🛠️ Requirements
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

## ▶️ How to Run
```bash
cd 5-airbnb-price-predictor
python analysis.py
```

## 🔍 Key Findings
- Manhattan listings are the most expensive on average
- "Entire home/apt" listings cost 2× more than private rooms
- Number of reviews has a weak negative correlation with price
- Random Forest achieves solid predictive accuracy

## 🧰 Libraries Used
`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`
