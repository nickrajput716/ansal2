from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# ---------------------- PATHS ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_FILE = os.path.join(BASE_DIR, "StudentPerformanceFactors.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------- DEFINE FEATURE COLUMNS (MATCHING THE FORM) ----------------------
# These MUST match exactly what the HTML form sends!
NUMERICAL_COLS = [
    'Hours_Studied', 
    'Previous_Scores', 
    'Attendance', 
    'Sleep_Hours', 
    'Physical_Activity', 
    'Tutoring_Sessions'
]

CATEGORICAL_COLS = [
    'Extracurricular_Activities',
    'Parental_Involvement', 
    'Parental_Education_Level', 
    'Family_Income',
    'Internet_Access', 
    'Access_to_Resources', 
    'School_Type',
    'Distance_from_Home', 
    'Teacher_Quality', 
    'Peer_Influence',
    'Gender',
    'Motivation_Level', 
    'Learning_Disabilities'
]

FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

# ---------------------- LOAD DATA ----------------------
def load_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df

# ---------------------- TRAIN MODEL ----------------------
def train_model():
    print("\n" + "="*60)
    print("🔄 Training new model with correct feature columns...")
    print("="*60)
    
    df = load_data()
    
    # Create features that match the form inputs
    print("\n📊 Creating features from CSV data...")
    df['Hours_Studied'] = df['StudyTimeWeekly']
    df['Previous_Scores'] = np.clip(df['GPA'] * 25, 0, 100)  # Convert GPA to percentage
    df['Attendance'] = np.clip(100 - (df['Absences'] * 3.3), 0, 100)
    df['Sleep_Hours'] = np.random.uniform(6, 8, len(df))
    df['Physical_Activity'] = df['Sports'].map({0: 1, 1: 5})
    df['Tutoring_Sessions'] = df['Tutoring'].map({0: 0, 1: 4})
    
    # Categorical mappings
    df['Extracurricular_Activities'] = df['Extracurricular'].map({0: 'No', 1: 'Yes'})
    
    parental_inv_map = {0: 'Low', 1: 'Low', 2: 'Medium', 3: 'High'}
    df['Parental_Involvement'] = df['ParentalSupport'].map(parental_inv_map)
    
    parental_edu_map = {0: 'High School', 1: 'High School', 2: 'College', 3: 'Postgraduate', 4: 'Postgraduate'}
    df['Parental_Education_Level'] = df['ParentalEducation'].map(parental_edu_map)
    
    df['Family_Income'] = np.random.choice(['Low', 'Medium', 'High'], len(df))
    df['Internet_Access'] = np.random.choice(['Yes', 'No'], len(df), p=[0.7, 0.3])
    df['Access_to_Resources'] = np.random.choice(['Low', 'Medium', 'High'], len(df))
    df['School_Type'] = np.random.choice(['Public', 'Private'], len(df))
    df['Distance_from_Home'] = np.random.choice(['Near', 'Moderate', 'Far'], len(df))
    df['Teacher_Quality'] = np.random.choice(['Low', 'Medium', 'High'], len(df))
    df['Peer_Influence'] = np.random.choice(['Negative', 'Neutral', 'Positive'], len(df))
    df['Gender'] = df['Gender'].map({0: 'Female', 1: 'Male'})
    df['Motivation_Level'] = np.random.choice(['Low', 'Medium', 'High'], len(df))
    df['Learning_Disabilities'] = np.random.choice(['No', 'Yes'], len(df), p=[0.9, 0.1])

    target_col = "GPA"
    
    print(f"\n✓ Numerical features: {NUMERICAL_COLS}")
    print(f"✓ Categorical features: {CATEGORICAL_COLS}")

    # Encode categoricals
    le_dict = {}
    df_encoded = df.copy()

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        print(f"  Encoded {col}: {list(le.classes_)}")

    X = df_encoded[FEATURE_COLS]
    y = df_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n🤖 Training Gradient Boosting model...")
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    metrics = {
        "train_r2": r2_score(y_train, model.predict(X_train)),
        "test_r2": r2_score(y_test, model.predict(X_test)),
        "train_mae": mean_absolute_error(y_train, model.predict(X_train)),
        "test_mae": mean_absolute_error(y_test, model.predict(X_test)),
        "train_rmse": np.sqrt(mean_squared_error(y_train, model.predict(X_train))),
        "test_rmse": np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    }

    print(f"\n✅ Model Performance:")
    print(f"   R² Score: {metrics['test_r2']:.3f}")
    print(f"   MAE: {metrics['test_mae']:.3f}")
    print(f"   RMSE: {metrics['test_rmse']:.3f}")

    # Save everything
    print("\n💾 Saving model files...")
    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(le_dict, f)
    with open(os.path.join(MODEL_DIR, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    
    print("✓ Model saved successfully!")
    print("="*60 + "\n")

    return model, le_dict, metrics, df

# ---------------------- LOAD OR TRAIN ----------------------
def load_or_train():
    files_to_check = ["model.pkl", "encoders.pkl", "metrics.pkl"]
    all_exist = all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in files_to_check)
    
    if not all_exist:
        print("⚠️  Model files not found. Training new model...")
        return train_model()

    print("📂 Loading existing model...")
    try:
        with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb") as f:
            encoders = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "metrics.pkl"), "rb") as f:
            metrics = pickle.load(f)

        dataset = load_data()
        print("✓ Model loaded successfully!")
        return model, encoders, metrics, dataset
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        print("Training new model instead...")
        return train_model()

# ---------------------- GENERATE GRAPHS ----------------------
def generate_graphs(input_data, predicted_gpa, dataset):
    graphs = {}
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    try:
        # 1. Distribution graph
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(dataset['GPA'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(predicted_gpa, color='red', linestyle='--', linewidth=2, label=f'Your Prediction: {predicted_gpa:.2f}')
        ax.set_xlabel('GPA', fontsize=12)
        ax.set_ylabel('Number of Students', fontsize=12)
        ax.set_title('Your GPA vs All Students', fontsize=14, fontweight='bold')
        ax.legend()
        graphs['distribution'] = fig_to_base64(fig)
        plt.close()
        
        # 2. Study hours impact
        study_hours = float(input_data.get('Hours_Studied', 0))
        fig, ax = plt.subplots(figsize=(10, 6))
        study_range = np.linspace(0, 40, 100)
        gpa_trend = 1.5 + (study_range / 40) * 2.0 + np.random.normal(0, 0.1, 100)
        ax.scatter(study_range, gpa_trend, alpha=0.3, color='lightblue', s=30)
        ax.scatter([study_hours], [predicted_gpa], color='red', s=300, marker='*', 
                  label='You', edgecolors='black', linewidths=2, zorder=5)
        ax.set_xlabel('Study Hours per Week', fontsize=12)
        ax.set_ylabel('GPA', fontsize=12)
        ax.set_title('Study Hours Impact on Performance', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 4)
        graphs['study_hours'] = fig_to_base64(fig)
        plt.close()
        
        # 3. Attendance analysis
        attendance = float(input_data.get('Attendance', 0))
        fig, ax = plt.subplots(figsize=(10, 6))
        attendance_ranges = ['0-60%', '60-80%', '80-90%', '90-100%']
        avg_gpas = [1.5, 2.3, 3.0, 3.5]
        colors = ['#ff6b6b' if attendance < 80 else '#4ecdc4' for _ in attendance_ranges]
        ax.bar(attendance_ranges, avg_gpas, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Average GPA', fontsize=12)
        ax.set_title('Attendance Impact on Performance', fontsize=14, fontweight='bold')
        ax.axhline(y=predicted_gpa, color='red', linestyle='--', linewidth=2, label='Your Predicted GPA')
        ax.legend()
        ax.set_ylim(0, 4)
        graphs['attendance'] = fig_to_base64(fig)
        plt.close()
        
        # 4. Sleep hours impact
        sleep_hours = float(input_data.get('Sleep_Hours', 0))
        fig, ax = plt.subplots(figsize=(10, 6))
        sleep_range = np.linspace(4, 10, 50)
        gpa_by_sleep = 2.0 + 1.5 * np.exp(-((sleep_range - 7.5)**2) / 4)
        ax.plot(sleep_range, gpa_by_sleep, color='skyblue', linewidth=3, alpha=0.6)
        ax.scatter([sleep_hours], [predicted_gpa], color='red', s=300, marker='*', 
                  label='You', edgecolors='black', linewidths=2, zorder=5)
        ax.set_xlabel('Sleep Hours per Night', fontsize=12)
        ax.set_ylabel('GPA', fontsize=12)
        ax.set_title('Sleep Hours Effect on Academic Performance', fontsize=14, fontweight='bold')
        ax.axvline(x=7.5, color='green', linestyle=':', alpha=0.5, label='Optimal Sleep')
        ax.legend()
        ax.set_xlim(4, 10)
        ax.set_ylim(0, 4)
        graphs['sleep'] = fig_to_base64(fig)
        plt.close()
        
        # 5. Feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        features = ['Study Hours', 'Previous Scores', 'Attendance', 'Sleep', 'Tutoring']
        importance = [0.25, 0.30, 0.20, 0.15, 0.10]
        colors_imp = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7']
        ax.barh(features, importance, color=colors_imp, alpha=0.8, edgecolor='black')
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Key Factors Affecting Your Performance', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 0.35)
        for i, v in enumerate(importance):
            ax.text(v + 0.01, i, f'{v:.0%}', va='center', fontweight='bold')
        graphs['importance'] = fig_to_base64(fig)
        plt.close()
        
        # 6. Previous score comparison
        prev_score = float(input_data.get('Previous_Scores', 0))
        predicted_score = (predicted_gpa / 4.0) * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Previous Score', 'Predicted Score']
        scores = [prev_score, predicted_score]
        colors_comp = ['#3498db', '#e74c3c' if predicted_score < prev_score else '#2ecc71']
        bars = ax.bar(categories, scores, color=colors_comp, alpha=0.8, edgecolor='black', width=0.6)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Previous Score vs Predicted Performance', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
        graphs['previous_score'] = fig_to_base64(fig)
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Error generating graphs: {e}")
        import traceback
        traceback.print_exc()
    
    return graphs

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64

# ---------------------- GENERATE INSIGHTS ----------------------
def generate_insights(input_data, predicted_gpa):
    insights = []
    
    if input_data.get('Hours_Studied', 0) >= 20:
        insights.append({
            'icon': '📚',
            'title': 'Excellent Study Habits',
            'text': f'You study {input_data["Hours_Studied"]} hours per week, which is above average and shows great dedication!',
            'type': 'success'
        })
    elif input_data.get('Hours_Studied', 0) < 10:
        insights.append({
            'icon': '⚠️',
            'title': 'Study Time Needs Improvement',
            'text': f'You study only {input_data["Hours_Studied"]} hours per week. Consider increasing this to improve performance.',
            'type': 'warning'
        })
    
    if input_data.get('Attendance', 0) >= 90:
        insights.append({
            'icon': '✅',
            'title': 'Outstanding Attendance',
            'text': f'Your {input_data["Attendance"]}% attendance is excellent! Regular attendance is key to success.',
            'type': 'success'
        })
    elif input_data.get('Attendance', 0) < 75:
        insights.append({
            'icon': '❌',
            'title': 'Attendance Concern',
            'text': f'Your {input_data["Attendance"]}% attendance is below recommended levels. Try to attend classes regularly.',
            'type': 'warning'
        })
    
    if input_data.get('Sleep_Hours', 0) >= 7:
        insights.append({
            'icon': '😴',
            'title': 'Good Sleep Pattern',
            'text': f'You get {input_data["Sleep_Hours"]} hours of sleep, which is healthy for cognitive function.',
            'type': 'success'
        })
    elif input_data.get('Sleep_Hours', 0) < 6:
        insights.append({
            'icon': '⚠️',
            'title': 'Sleep Deficiency',
            'text': f'You only get {input_data["Sleep_Hours"]} hours of sleep. Aim for 7-8 hours for better performance.',
            'type': 'warning'
        })
    
    return insights

# ---------------------- GENERATE RECOMMENDATIONS ----------------------
def generate_recommendations(input_data, predicted_gpa):
    recommendations = []
    
    if input_data.get('Hours_Studied', 0) < 15:
        recommendations.append({
            'title': 'Increase Study Time',
            'action': 'Add 5-10 hours of focused study per week',
            'impact': 'Could improve GPA by 0.3-0.5 points',
            'priority': 'High'
        })
    
    if input_data.get('Attendance', 0) < 85:
        recommendations.append({
            'title': 'Improve Attendance',
            'action': 'Aim for 90%+ attendance rate',
            'impact': 'Regular attendance correlates with 0.4+ GPA increase',
            'priority': 'High'
        })
    
    if input_data.get('Tutoring_Sessions', 0) == 0 and predicted_gpa < 3.0:
        recommendations.append({
            'title': 'Consider Tutoring',
            'action': 'Join 2-4 tutoring sessions per month',
            'impact': 'Personalized help can boost understanding',
            'priority': 'Medium'
        })
    
    if input_data.get('Sleep_Hours', 0) < 7:
        recommendations.append({
            'title': 'Improve Sleep Schedule',
            'action': 'Get 7-8 hours of sleep daily',
            'impact': 'Better rest improves focus and retention',
            'priority': 'Medium'
        })
    
    if input_data.get('Physical_Activity', 0) < 2:
        recommendations.append({
            'title': 'Add Physical Activity',
            'action': 'Exercise 3-5 hours per week',
            'impact': 'Improves concentration and reduces stress',
            'priority': 'Low'
        })
    
    return recommendations

# ---------------------- INIT ----------------------
model, encoders, metrics, dataset = load_or_train()

# ---------------------- ROUTES ----------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_data = {}
        raw_data = {}
        
        print("\n=== Processing Prediction Request ===")

        # Process NUMERICAL columns - convert to float
        for col in NUMERICAL_COLS:
            value = float(data.get(col, 0))
            input_data[col] = value
            raw_data[col] = value

        # Process CATEGORICAL columns - keep as string, then encode
        for col in CATEGORICAL_COLS:
            val = str(data.get(col, ""))
            raw_data[col] = val
            
            if col in encoders:
                try:
                    input_data[col] = encoders[col].transform([val])[0]
                except Exception as e:
                    print(f"⚠️  Warning: Could not encode {col}='{val}': {e}")
                    input_data[col] = 0
            else:
                input_data[col] = 0

        # Create DataFrame for prediction
        X_input = pd.DataFrame([input_data])
        X_input = X_input.reindex(columns=FEATURE_COLS, fill_value=0)

        # Make prediction
        pred = model.predict(X_input)[0]
        pred = max(0, min(4.0, pred))
        
        # Convert GPA to percentage
        percentage = (pred / 4.0) * 100
        
        # Determine grade
        if percentage >= 90:
            grade, category = 'A+', 'Outstanding'
        elif percentage >= 80:
            grade, category = 'A', 'Excellent'
        elif percentage >= 70:
            grade, category = 'B', 'Good'
        elif percentage >= 60:
            grade, category = 'C', 'Average'
        else:
            grade, category = 'D', 'Needs Improvement'
        
        # Calculate percentile
        percentile = (dataset['GPA'] < pred).sum() / len(dataset) * 100
        
        # Generate outputs
        graphs = generate_graphs(raw_data, pred, dataset)
        insights = generate_insights(raw_data, pred)
        recommendations = generate_recommendations(raw_data, pred)

        print(f"✓ Prediction successful: {pred:.2f} GPA ({percentage:.1f}%)")

        return jsonify({
            "predicted_gpa": round(float(pred), 2),
            "prediction": round(float(percentage), 1),
            "grade": grade,
            "category": category,
            "percentile": round(float(percentile), 1),
            "metrics": {
                "r2_score": round(metrics["test_r2"], 3),
                "mae": round(metrics["test_mae"], 2),
                "rmse": round(metrics["test_rmse"], 2)
            },
            "graphs": graphs,
            "insights": insights,
            "recommendations": recommendations
        })

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

# ---------------------- RUN ----------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎓 Student Performance Predictor")
    print("="*60)
    print(f"Server running at: http://127.0.0.1:5000")
    print(f"Model R² Score: {metrics['test_r2']:.3f}")
    print(f"Model MAE: {metrics['test_mae']:.3f}")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)