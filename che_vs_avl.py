import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pd.options.display.max_columns = None
df=pd.read_csv('epl_matches_2023_2025.csv')

#function to extract meaningful data only
def clean_data(data):
    cleaned_data = data[['homeTeam.name', 'awayTeam.name', 'score.fullTime.home', 'score.fullTime.away']].copy()
    # Extract season year from 'season.startDate'
    season_year = int(str(data['season.startDate'])[:4]) if pd.notnull(data['season.startDate']) else None
    cleaned_data['season'] = season_year
    if cleaned_data['score.fullTime.home'] > cleaned_data['score.fullTime.away']:
        cleaned_data['result'] = 'H'
    elif cleaned_data['score.fullTime.home'] < cleaned_data['score.fullTime.away']:
        cleaned_data['result'] = 'A'
    else:
        cleaned_data['result'] = 'D'
    return cleaned_data
matches = df.apply(clean_data, axis=1)

# Function to get last 5 matches stats for a team
def get_team_last5_stats(df, idx, team, season):
    past = df.iloc[:idx]

    # restrict to same season only
    past = past[past["season"] == season]

    team_matches = past[
        (past["homeTeam.name"] == team) |
        (past["awayTeam.name"] == team)
    ].tail(5)

    if team_matches.empty:
        return 0, 0.0, 0.0

    # goals scored
    goals_for = np.where(
        team_matches["homeTeam.name"] == team,
        team_matches["score.fullTime.home"],
        team_matches["score.fullTime.away"]
    )

    # goals conceded
    goals_against = np.where(
        team_matches["homeTeam.name"] == team,
        team_matches["score.fullTime.away"],
        team_matches["score.fullTime.home"]
    )

    # points
    points = np.where(
        goals_for > goals_against, 3,
        np.where(goals_for == goals_against, 1, 0)
    )

    return points.sum(), goals_for.mean(), goals_against.mean()

# Generate features for each match
features = []

for i, row in matches.iterrows():
    home_form, home_gf, home_ga = get_team_last5_stats(
        matches, i, row["homeTeam.name"], row["season"]
    )

    away_form, away_gf, away_ga = get_team_last5_stats(
        matches, i, row["awayTeam.name"], row["season"]
    )

    features.append({
        **row,
        "home_form_5": home_form,
        "home_avg_gf_5": home_gf,
        "home_avg_ga_5": home_ga,
        "away_form_5": away_form,
        "away_avg_gf_5": away_gf,
        "away_avg_ga_5": away_ga
    })

new_df = pd.DataFrame(features)

result_map = {
    "H": 0,   # Home win
    "A": 1,   # Away win
    "D": 2    # Draw
}

new_df["result_encoded"] = new_df["result"].map(result_map)
new_df[["result", "result_encoded"]].tail(10)

X = new_df.drop(columns=[
    "result",          # original label
    "result_encoded",  # encoded label
    "homeTeam.name",   # categorical text
    "awayTeam.name", # categorical text
    "score.fullTime.home",
    "score.fullTime.away",
    "season"
])

#dropping the first 50 rows as stats are not fully computed
X=X.iloc[51:]
y = new_df["result_encoded"].iloc[51:]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y 
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=7,
    max_features="sqrt",
    class_weight="balanced",
    random_state=39
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("confusion matrix \n 0 = home win, 1 = away win, 2 = draw \n\n", confusion_matrix(y_test, y_pred))
print("\n\n","================= Accuracy ================\n\n")
print("accuracy compared to test data: ", accuracy_score(y_test, y_pred), "\n\n")

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print("\n\n ========================== Feature importance ========================= \n\n",importance.head(10))

#check for overfitting
# train_pred = model.predict(X_train)
# test_pred  = model.predict(X_test)
# print("Train Acc:", accuracy_score(y_train, train_pred))
# print("Test  Acc:", accuracy_score(y_test, test_pred))

# Predicting Chelsea vs Aston Villa match on sunday
che_form, che_goals_for, che_goals_against = get_team_last5_stats(matches, 927, 'Chelsea FC', 2025)
avl_form, avl_goals_for, avl_goals_against = get_team_last5_stats(matches, 927, 'Aston Villa FC', 2025 )

X_new = pd.DataFrame(
    [[che_form, che_goals_for, che_goals_against, avl_form, avl_goals_for, avl_goals_against]],
    columns=X_train.columns
)
predicted_outcome = model.predict(X_new)[0]
if predicted_outcome == 0:
    outcome = "Chelsea Win"
elif predicted_outcome == 1:
    outcome = "Aston Villa Win"
else:
    outcome = "Draw"
print("\n\n The predicted outcome for Chelsea vs Aston Villa is:", outcome)

# Predict probabilities
probs = model.predict_proba(X_new)
print("\n\n Predicted probabilities for match outcomes (Chelsea, Aston Villa, Draw):", probs[0])
