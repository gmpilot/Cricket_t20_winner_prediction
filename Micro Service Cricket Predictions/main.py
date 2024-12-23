import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean data
matches_df = pd.read_csv("matches.csv")
deliveries_df = pd.read_csv("deliveries.csv")

matches_df = matches_df.drop(
    [
        "Season",
        "date",
        "toss_winner",
        "toss_decision",
        "result",
        "dl_applied",
        "player_of_match",
        "venue",
        "umpire1",
        "umpire2",
        "umpire3",
    ],
    axis=1,
)
matches_df = matches_df.rename(columns={"city": "stadium_city", "id": "match_id"})

deliveries_df = deliveries_df[
    ["match_id", "batting_team", "total_runs", "extra_runs", "player_dismissed", "over"]
]
deliveries_df["total_wickets"] = deliveries_df["player_dismissed"].notna().astype(int)

# Group data by match and team to calculate runs, balls, and wickets
match_team_stats = (
    deliveries_df.groupby(["match_id", "batting_team"])
    .agg(
        total_runs=("total_runs", "sum"),
        total_balls=("over", "count"),
        total_wickets=("player_dismissed", lambda x: x.notna().sum()),
    )
    .reset_index()
)

match_team_stats["total_overs"] = match_team_stats["total_balls"] / 6
match_team_stats["total_overs"] = match_team_stats["total_overs"].apply(
    lambda x: min(x, 20)
)

merged_df = pd.merge(matches_df, match_team_stats, on="match_id")
merged_df["runrate"] = merged_df["total_runs"] / merged_df["total_overs"]

# Split data for both teams
team1_data = merged_df[merged_df["batting_team"] == merged_df["team1"]].copy()
team2_data = merged_df[merged_df["batting_team"] == merged_df["team2"]].copy()

team1_data = team1_data.rename(
    columns={
        "total_runs": "Team1_runs",
        "total_wickets": "Team1_wickets",
        "runrate": "Team1_runrate",
        "total_overs": "Team1_overs",
    }
).drop(columns=["batting_team", "total_balls"])

team2_data = team2_data.rename(
    columns={
        "total_runs": "Team2_runs",
        "total_wickets": "Team2_wickets",
        "runrate": "Team2_runrate",
        "total_overs": "Team2_overs",
    }
).drop(columns=["batting_team", "total_balls"])

consolidated_df = pd.merge(team1_data, team2_data, on="match_id")

consolidated_df = consolidated_df.drop(
    columns=[
        "stadium_city_y",
        "team1_y",
        "team2_y",
        "winner_y",
        "win_by_runs_y",
        "win_by_wickets_y",
    ]
)
consolidated_df = consolidated_df.rename(
    columns={
        "stadium_city_x": "City",
        "team1_x": "Team1",
        "team2_x": "Team2",
        "winner_x": "Winner",
        "win_by_runs_x": "Win_by_runs",
        "win_by_wickets_x": "Win_by_wickets",
    }
)


# Calculate run rate difference
def calculate_runrate_difference(row):
    if row["Winner"] == row["Team1"]:
        return row["Team1_runrate"] - row["Team2_runrate"]
    else:
        return row["Team2_runrate"] - row["Team1_runrate"]


consolidated_df["runrate_difference"] = consolidated_df.apply(
    calculate_runrate_difference, axis=1
)

final_df = consolidated_df[
    [
        "City",
        "Team1",
        "Team1_runs",
        "Team1_wickets",
        "Team1_runrate",
        "Team1_overs",
        "Team2",
        "Team2_runs",
        "Team2_wickets",
        "Team2_runrate",
        "Team2_overs",
        "Winner",
        "Win_by_runs",
        "Win_by_wickets",
        "runrate_difference",
    ]
]

# Prepare features and target
X = final_df[
    [
        "Team1_runs",
        "Team1_wickets",
        "Team1_overs",
        "Team2_runs",
        "Team2_wickets",
        "Team2_overs",
        "runrate_difference",
    ]
]
y = final_df.apply(lambda row: 1 if row["Winner"] == row["Team1"] else 0, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


# Predict winner based on team stats
def predict_winner(first_batting_stats, second_batting_stats, target_runs):
    first_batting_runrate = first_batting_stats["runs"] / first_batting_stats["overs"]
    second_batting_runrate = (
        second_batting_stats["runs"] / second_batting_stats["overs"]
    )

    runrate_diff = first_batting_runrate - second_batting_runrate

    prediction_data = pd.DataFrame(
        [
            {
                "Team1_runs": first_batting_stats["runs"],
                "Team1_wickets": first_batting_stats["wickets"],
                "Team1_overs": first_batting_stats["overs"],
                "Team2_runs": second_batting_stats["runs"],
                "Team2_wickets": second_batting_stats["wickets"],
                "Team2_overs": second_batting_stats["overs"],
                "runrate_difference": runrate_diff,
            }
        ]
    )

    winner_pred = model.predict(prediction_data)

    if second_batting_stats["runs"] < target_runs:
        remaining_runs = target_runs - second_batting_stats["runs"]
        remaining_overs = 20 - second_batting_stats["overs"]
        remaining_balls = remaining_overs * 6

        required_runrate = remaining_runs / remaining_overs
        if required_runrate > 36 or remaining_runs > remaining_balls * 6:
            return "Team1"

        if second_batting_runrate >= required_runrate:
            return "Team2"

    return "Team1" if winner_pred == 1 else "Team2"


# GUI Implementation
def show_accuracy():
    footer_output.config(state="normal")
    footer_output.delete(1.0, tk.END)
    footer_output.insert(
        tk.END,
        f"Accuracy: {accuracy}\n\nConfusion Matrix:\n{conf_matrix}\n\nClassification Report:\n{class_report}",
    )
    footer_output.config(state="disabled")


def predict_gui():
    first_batting_stats = {
        "runs": int(entry_runs_team1.get()),
        "wickets": int(entry_wickets_team1.get()),
        "overs": float(entry_overs_team1.get()),
    }

    second_batting_stats = {
        "runs": int(entry_runs_team2.get()),
        "wickets": int(entry_wickets_team2.get()),
        "overs": float(entry_overs_team2.get()),
    }

    target_runs = int(entry_target_runs.get())

    predicted_winner = predict_winner(
        first_batting_stats, second_batting_stats, target_runs
    )

    footer_output.config(state="normal")
    footer_output.delete(1.0, tk.END)
    footer_output.insert(tk.END, f"The predicted winner is: {predicted_winner}")
    footer_output.config(state="disabled")


# Visualize Plot with Run Rate and Required Run Rate for Winning
def visualize_plot():
    first_batting_stats = {
        "runs": int(entry_runs_team1.get()),
        "wickets": int(entry_wickets_team1.get()),
        "overs": float(entry_overs_team1.get()),
    }

    second_batting_stats = {
        "runs": int(entry_runs_team2.get()),
        "wickets": int(entry_wickets_team2.get()),
        "overs": float(entry_overs_team2.get()),
    }

    target_runs = int(entry_target_runs.get())

    remaining_runs = target_runs - second_batting_stats["runs"]
    remaining_overs = 20 - second_batting_stats["overs"]

    required_runrate = remaining_runs / remaining_overs if remaining_overs > 0 else 0

    plt.figure(figsize=(10, 5))

    sns.lineplot(
        x=[first_batting_stats["overs"], first_batting_stats["overs"]],
        y=[0, first_batting_stats["runs"]],
        label="Team1 Runs vs Overs",
        color="#1877F2",
        marker="o",
    )

    sns.lineplot(
        x=[second_batting_stats["overs"], second_batting_stats["overs"]],
        y=[0, second_batting_stats["runs"]],
        label="Team2 Runs vs Overs",
        color="#42B72A",
        marker="o",
    )

    plt.plot(
        [second_batting_stats["overs"], 20],
        [second_batting_stats["runs"], target_runs],
        "r--",
        label=f"Required Runrate: {required_runrate:.2f} runs/over",
    )

    current_runrate = (
        second_batting_stats["runs"] / second_batting_stats["overs"]
        if second_batting_stats["overs"] > 0
        else 0
    )
    plt.plot(
        [second_batting_stats["overs"], 20],
        [
            second_batting_stats["runs"],
            second_batting_stats["runs"]
            + (20 - second_batting_stats["overs"]) * current_runrate,
        ],
        "b-",
        label=f"Current Runrate: {current_runrate:.2f} runs/over",
    )

    predicted_runs = second_batting_stats["runs"]
    predicted_run_line = []
    for i in range(int(second_batting_stats["overs"]), 21):
        predicted_runs += current_runrate
        predicted_run_line.append(predicted_runs)

    plt.plot(
        range(int(second_batting_stats["overs"]), 21),
        predicted_run_line,
        "g--",
        label="Predicted Run Line",
    )

    plt.title("Run Rate vs Overs for Cricket Match Prediction")
    plt.xlabel("Overs")
    plt.ylabel("Runs")
    plt.legend()
    plt.show()


# GUI Setup
root = tk.Tk()
root.title("Cricket (T20) Match Prediction")

# Inputs
tk.Label(root, text="Team 1 Runs:").grid(row=0, column=0)
entry_runs_team1 = tk.Entry(root)
entry_runs_team1.grid(row=0, column=1)

tk.Label(root, text="Team 1 Wickets:").grid(row=1, column=0)
entry_wickets_team1 = tk.Entry(root)
entry_wickets_team1.grid(row=1, column=1)

tk.Label(root, text="Team 1 Overs:").grid(row=2, column=0)
entry_overs_team1 = tk.Entry(root)
entry_overs_team1.grid(row=2, column=1)

tk.Label(root, text="Team 2 Runs:").grid(row=3, column=0)
entry_runs_team2 = tk.Entry(root)
entry_runs_team2.grid(row=3, column=1)

tk.Label(root, text="Team 2 Wickets:").grid(row=4, column=0)
entry_wickets_team2 = tk.Entry(root)
entry_wickets_team2.grid(row=4, column=1)

tk.Label(root, text="Team 2 Overs:").grid(row=5, column=0)
entry_overs_team2 = tk.Entry(root)
entry_overs_team2.grid(row=5, column=1)

tk.Label(root, text="Target Runs:").grid(row=6, column=0)
entry_target_runs = tk.Entry(root)
entry_target_runs.grid(row=6, column=1)

# Buttons
tk.Button(root, text="Predict Winner", command=predict_gui).grid(
    row=7, column=0, columnspan=2
)
tk.Button(root, text="Show Accuracy", command=show_accuracy).grid(
    row=8, column=0, columnspan=2
)
tk.Button(root, text="Visualize Plot", command=visualize_plot).grid(
    row=9, column=0, columnspan=2
)

# Footer Output
footer_output = tk.Text(root, height=10, width=60, wrap=tk.WORD, state="disabled")
footer_output.grid(row=10, column=0, columnspan=2)

root.mainloop()
