
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from dotenv import load_dotenv
import os
load_dotenv()
API = os.getenv('API_KEY')


seasonURL={
    '2023':'https://api.football-data.org/v4/competitions/PL/matches/?season=2023',
    '2024':'https://api.football-data.org/v4/competitions/PL/matches/?season=2024',
    '2025':'https://api.football-data.org/v4/competitions/PL/matches/?season=2025&status=FINISHED'
}
headers = { 'X-Auth-Token': API }
full_matches = []
for season, uri in seasonURL.items():
    response = requests.get(uri, headers=headers)
    for match in response.json()['matches']:
        full_matches.append(match)

full_matches_df = pd.json_normalize(full_matches)

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

matches = full_matches_df.apply(clean_data, axis=1)

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

#function to get h2h stats for home and away team in last 5 matches
def get_h2h_last3_stats(df, idx, home_team, away_team):
    past = df.iloc[:idx]

    h2h_matches = past[
        ((past["homeTeam.name"] == home_team) & (past["awayTeam.name"] == away_team)) |
        ((past["homeTeam.name"] == away_team) & (past["awayTeam.name"] == home_team))
    ].tail(3)

    if h2h_matches.empty:
        return 0., 0.0, 0.0, 0.0

    # goals scored by home team
    goals_for = np.where(
        h2h_matches["homeTeam.name"] == home_team,
        h2h_matches["score.fullTime.home"],
        h2h_matches["score.fullTime.away"]
    )

    #goals scored by away team
    goals_for_away = np.where(
        h2h_matches["homeTeam.name"] == away_team,
        h2h_matches["score.fullTime.home"],
        h2h_matches["score.fullTime.away"]
    )

    # goals conceded by home team
    goals_against = np.where(
        h2h_matches["homeTeam.name"] == home_team,
        h2h_matches["score.fullTime.away"],
        h2h_matches["score.fullTime.home"]
    )

    #goals conceded by away team
    goals_against_away = np.where(
        h2h_matches["homeTeam.name"] == away_team,
        h2h_matches["score.fullTime.away"],
        h2h_matches["score.fullTime.home"]
    )   

    # points for home team
    home_points = np.where(
        goals_for > goals_against, 3,
        np.where(goals_for == goals_against, 1, 0)
    )

    #points for away team
    away_points = np.where(
        goals_for_away > goals_against_away, 3,
        np.where(goals_for_away == goals_against_away, 1, 0)
    )   
    return goals_for.mean(), goals_against.mean(), goals_for_away.mean(), goals_against_away.mean()

# Create new features for each match
features = []     
for idx, row in matches.iterrows():
    home_team = row['homeTeam.name']
    away_team = row['awayTeam.name']
    season = row['season']

    home_points, home_goals_for, home_goals_against = get_team_last5_stats(matches, idx, home_team, season)
    away_points, away_goals_for, away_goals_against = get_team_last5_stats(matches, idx, away_team, season)
    h2h_goals_for, h2h_goals_against, h2h_away_goals_for, h2h_away_goals_against = get_h2h_last3_stats(matches, idx, home_team, away_team)

    form_diff_last5 = home_points - away_points
    h2h_goals_scored_diff = h2h_goals_for - h2h_away_goals_for
    h2h_goals_conceded_diff = h2h_goals_against - h2h_away_goals_against


    features.append({
        **row.to_dict(),
        'h2h_goals_for': h2h_goals_for,
        'form_diff_last5': form_diff_last5,
        'home_goals_for_last5': home_goals_for,
        'away_goals_for_last5': away_goals_for,
        'home_goals_against_last5': home_goals_against,
        'away_goals_against_last5': away_goals_against,

        'h2h_goals_scored_diff': h2h_goals_scored_diff,
        'h2h_goals_conceded_diff': h2h_goals_conceded_diff,

    })
new_df = pd.DataFrame(features)

result_map = {
    "H": 0,   # Home win
    "A": 1,   # Away win
    "D": 2    # Draw
}
new_df["result_encoded"] = new_df["result"].map(result_map)

X = new_df[['form_diff_last5',
            'home_goals_for_last5',
            'away_goals_for_last5',
            'home_goals_against_last5',
            'away_goals_against_last5',

            'h2h_goals_scored_diff',
            'h2h_goals_conceded_diff']]
y = new_df['result_encoded']

X = X.iloc[51:]
y = y.iloc[51:]

X_train_unshuffled, X_test_unshuffled, y_train_unshuffled, y_test_unshuffled = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=5,
    max_features=0.8,
    max_samples=1.0,
    class_weight="balanced",
    random_state=39
)

model.fit(X_train_unshuffled, y_train_unshuffled)
y_pred = model.predict(X_test_unshuffled)
# Evaluation
print("Classification Report:\n\n", classification_report(y_test_unshuffled, y_pred))
print("confusion matrix \n 0 = home win, 1 = away win, 2 = draw \n\n", confusion_matrix(y_test_unshuffled, y_pred))
print("\n\n","================= Accuracy ================\n\n")
print("accuracy compared to test data: ", accuracy_score(y_test_unshuffled, y_pred), "\n\n")

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print("\n\n ========================== Feature importance ========================= \n\n",importance.head(20))

# #check for overfitting
# train_pred = model.predict(X_train_unshuffled)
# test_pred  = model.predict(X_test_unshuffled)
# print("Train Acc:", accuracy_score(y_train_unshuffled, train_pred))
# print("Test  Acc:", accuracy_score(y_test_unshuffled, test_pred))

# Predicting New vs Cry match on sunday
new_form, new_goals_for, new_goals_against = get_team_last5_stats(matches, len(matches)-1, 'Newcastle United FC', 2025)
cry_form, cry_goals_for, cry_goals_against = get_team_last5_stats(matches, len(matches)-1, 'Crystal Palace FC', 2025 )

difference = new_form - cry_form
h2h_new_goals_scored, h2h_new_goals_conceded, h2h_cry_goals_scored, h2h_cry_goals_conceded = get_h2h_last3_stats(matches, len(matches)-1, 'Newcastle United FC', 'Crystal Palace FC')
h2h_goals_scored_diff_match = h2h_new_goals_scored - h2h_cry_goals_scored
h2h_goals_conceded_diff_match =  h2h_new_goals_conceded - h2h_cry_goals_conceded

X_new = pd.DataFrame(
    [[difference, new_goals_for, cry_goals_for, new_goals_against, cry_goals_against, h2h_goals_scored_diff_match, h2h_goals_conceded_diff_match]],
    columns=X_train_unshuffled.columns
)
predicted_outcome = model.predict(X_new)[0]
if predicted_outcome == 0:
    outcome = "Newcastle United Win"
elif predicted_outcome == 1:
    outcome = "Crystal Palace Win"
else:
    outcome = "Draw"
print("\n\n The predicted outcome for Newcastle United vs Crystal Palace is:", outcome)

# Predict probabilities
probs = model.predict_proba(X_new)
print("\n\n Predicted probabilities for match outcomes (Newcastle United, Crystal Palace, Draw):", probs[0])
