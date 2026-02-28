from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
data = pd.read_excel("crop_recommendation_1000.xlsx")

max_water = data["Water_Requirement_mm"].max()

# Create recommendation column
data["Recommendation"] = (
    data["Crop"] + " | " +
    data["Variety"] + " | " +
    data["Duration_Days"].astype(str) + " days | " +
    data["Fertilizer"] + " | " +
    data["Water_Requirement_mm"].astype(str)
)

# Label Encoding
le_soil = LabelEncoder()
le_season = LabelEncoder()
le_target = LabelEncoder()

data["Soil_Type"] = le_soil.fit_transform(data["Soil_Type"])
data["Season"] = le_season.fit_transform(data["Season"])
data["Recommendation"] = le_target.fit_transform(data["Recommendation"])

X = data[["Soil_Type", "Season", "pH_Min", "pH_Max"]]
y = data["Recommendation"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    soil = request.form["soil"]
    season = request.form["season"]

    # ✅ pH Validation (Backend Security)
    try:
        ph = float(request.form["ph"])
    except:
        return "Invalid pH value"

    if ph < 5.5 or ph > 7.7:
        return "pH value is wrong. Allowed range: 5.5 - 7.7"

    # Encoding
    soil_encoded = le_soil.transform([soil])[0]
    season_encoded = le_season.transform([season])[0]

    # Prediction
    prediction = model.predict([[soil_encoded, season_encoded, ph, ph]])
    result = le_target.inverse_transform(prediction)[0]

    parts = result.split(" | ")

    water_mm = float(parts[4])
    water_percent = round((water_mm / max_water) * 100, 2)

    return render_template(
        "result.html",
        crop=parts[0],
        variety=parts[1],
        duration=parts[2],
        fertilizer=parts[3],
        water_percent=water_percent
    )


if __name__ == "__main__":
    app.run(debug=True)
