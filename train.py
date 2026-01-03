import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

# 1. LOAD YOUR DATA
# We load the CSV you just collected!
df = pd.read_csv('survey_results.csv')


print("\nCOLUMNS IN CSV:\n")
for col in df.columns:
    print(col)

df = df.rename(columns={
    "I have a vivid imagination and enjoy hearing new, abstract ideas.": "Openness",
    "I am curious about many different things and like trying new experiences.": "Openness_2",

    "I am always prepared, organized, and follow a schedule": "Conscientiousness",
    "I am diligent and complete tasks thoroughly and on time.": "Conscientiousness_2",

    " I am the life of the party and feel energized by social interactions. ": "Extraversion",
    " I enjoy being the center of attention and meeting new people.": "Extraversion_2",

    " I sympathize with others' feelings and try to make people feel at ease.": "Agreeableness",
    "I am generally trusting and kind to almost everyone I meet.": "Agreeableness_2",

    " I get stressed out easily and frequently worry about things. ": "Neuroticism",
    " I often feel blue or experience frequent mood swings.": "Neuroticism_2",

    " Which of the following study techniques have you used and found highly effective for your learning? (Select all that apply)": "Study_Techniques"
})

df["Openness"] = (df["Openness"].astype(int) + df["Openness_2"].astype(int)) / 2
df["Conscientiousness"] = (df["Conscientiousness"].astype(int) + df["Conscientiousness_2"].astype(int)) / 2
df["Extraversion"] = (df["Extraversion"].astype(int) + df["Extraversion_2"].astype(int)) / 2
df["Agreeableness"] = (df["Agreeableness"].astype(int) + df["Agreeableness_2"].astype(int)) / 2
df["Neuroticism"] = (df["Neuroticism"].astype(int) + df["Neuroticism_2"].astype(int)) / 2

techniques = [
    "Spaced Repetition",
    "Active Recall",
    "Feynman Technique",
    "Mind Mapping",
    "Pomodoro Technique"
]

for tech in techniques:
    df[tech] = df["Study_Techniques"].fillna("").str.contains(tech).astype(int)


# 2. IDENTIFY YOUR COLUMNS
# Note: Ensure these column names match your CSV headers exactly!
# Features (Personality)
feature_cols = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
# Targets (Study Techniques)
target_cols = ['Spaced Repetition', 'Active Recall', 'Feynman Technique', 'Mind Mapping', 'Pomodoro Technique']

X = df[feature_cols]
y = df[target_cols]

# 3. SPLIT & SCALE
# We keep 20% of data to test if the model actually learned
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. TRAIN THE BRAIN
# We use Random Forest - it's great for smaller university datasets
model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100))
model.fit(X_train_scaled, y_train)

# 5. SAVE THE ASSETS
# These are the files your Web App/UI will "read" later
with open('personality_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(" Success, Arpita Lakhisirani! Your model has been trained.")
print("Check your folder for 'personality_model.pkl' and 'scaler.pkl'.")