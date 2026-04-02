# Train an SVM model to predict NBA game outcomes using
# last-season averages, current-season momentum, and Elo ratings.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import joblib
import boto3, io, os

from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nba_api.stats.library.http import NBAStatsHTTP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


# ------------------
# HTTP / SESSION SETUP
# sets up a shared HTTP session for making calls to nba_api
# designed to avoid rate limits and retry failed requests when calling nba_api endpoints
def configure_nba_session(timeout: int = 20) -> None:
    # configure a shared NBAStatsHTTP session with retries so we don't
    # get rate-limited or break on transient errors.
    # create session
    s = Session()
    # define retry behavior - try six times to read and connect,
    # wait longer between each retry
    # retry if status code is one of the following,
    # and wont raise flag on exception failures
    retry = Retry(
        total=6,
        read=6,
        connect=6,
        backoff_factor=0.7,
        status_forcelist=[429, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    # attach retry behavior to all HTTP sessions
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    # set header to mimic a browser, make request to look like its coming from a browser
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            "Referer": "https://www.nba.com/",
            "Origin": "https://www.nba.com",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
    )
    # set the session in nba_api internal client
    # tells the nba api to use shared session,
    # apply and retry header settings,
    # and use our custom timeout
    NBAStatsHTTP.timeout = timeout
    NBAStatsHTTP._session = s


# ------------------
# DATA LOADING & BASIC FEATURES

# pulls game logs from the nba api for a given season
# returns a cleaned and formatted pandas dataframe, one row per team per game
# each row corresponds to a teams performance in a specific game
def load_season_team_logs(season_label: str) -> pd.DataFrame:
    print(f"Fetching team game logs for season {season_label} ...")
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    bucket = os.environ.get("S3_BUCKET", "statline-nba-data")
    season_safe = season_label.replace("-", "_")
    key = f"game_logs/season_{season_safe}.csv"
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.StringIO(obj["Body"].read().decode("utf-8")))
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["SEASON"] = season_label
    df["NET_MARGIN"] = df["PLUS_MINUS"]
    df["WIN"] = (df["WL"] == "W").astype(int)
    return df

# calculates the per team avaerage stats from the previous nba season
# takes the dataframe where each row represents one teams performance in one game from last season
def compute_last_season_team_averages(df_last: pd.DataFrame) -> pd.DataFrame:
    # Compute last-season per-team averages for key stats.
    print("Computing last-season per-team averages ...")
    # group input datafrsme by team id and compute the mean of the key stats
    agg = (
        df_last.groupby("TEAM_ID")
        .agg(
            FG_PCT=("FG_PCT", "mean"),
            FG3_PCT=("FG3_PCT", "mean"),
            FT_PCT=("FT_PCT", "mean"),
            PTS=("PTS", "mean"),
            REB=("REB", "mean"),
            AST=("AST", "mean"),
            TOV=("TOV", "mean"),
            PLUS_MINUS=("PLUS_MINUS", "mean"),
            NET_MARGIN=("NET_MARGIN", "mean"),
        )
        .reset_index()
    )
    # return resulting data frame with the per team avaeraged stats
    return agg

# ------------------
# ELO RATING

# calculates the ELO ratings for nba teams over time and adds pre game elo rating to each row in the data set
# Elo is rating system commonly used in games and sports to estimate the relative skill level of players or teams
# takes in all game data and computes elo rating


def add_elo_column(
    df_all: pd.DataFrame, base_elo: float = 1500.0
) -> pd.DataFrame:
    # Compute an Elo rating for each team over time and attach ELO_PRE
    # (rating before the game) to each row in df_all.
    # We process games in chronological order across seasons.
    print("Computing Elo ratings across last + current season ...")
    # sort dataset in chronological order by game date and id
    df = df_all.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    # initialize a dictionary to track latest elo rating for each team
    team_elo: Dict[int, float] = {}
    # create a list to store each team pre game elo for every row
    elo_pre_list: list[float] = []
    # loop through each game by grouping all rows with same game_id
    for game_id, g in df.groupby("GAME_ID", sort=False):
        # expect two rows (two teams), skip weird stuff, if game doesnt have two teams append base elo
        if len(g) != 2:
            for _ in range(len(g)):
                elo_pre_list.append(base_elo)
            continue
        # convert grouped game rows to named tuples and get team ids
        rows = list(g.itertuples(index=False))
        t1 = int(getattr(rows[0], "TEAM_ID"))
        t2 = int(getattr(rows[1], "TEAM_ID"))
        # fetch each teams current elo, if not set use base elo 1500
        elo1 = team_elo.get(t1, base_elo)
        elo2 = team_elo.get(t2, base_elo)

        # calculate expected win probability for both teams using elo formula
        e1 = 1.0 / (1.0 + 10 ** ((elo2 - elo1) / 400))
        e2 = 1.0 - e1
        # assign game outcome, 1 for win 0 for loss
        w1 = 1.0 if getattr(rows[0], "WL") == "W" else 0.0
        w2 = 1.0 if getattr(rows[1], "WL") == "W" else 0.0

        # determine k factor, how aggresively the elo changes based on how far along the season is
        # determines how much a teams rating changes after a game
        idx = len(team_elo)
        if idx < 300:
            k = 20
        elif idx < 800:
            k = 15
        else:
            k = 10
        # update the elo ratings based on the difference between expected and actual outcome
        # unexpected wins , elo increases
        # expected win, elo increases a little
        # loses unexpected, elo drops
        new_elo1 = elo1 + k * (w1 - e1)
        new_elo2 = elo2 + k * (w2 - e2)

        # store the elo rating before the game for both teams and save updated elo
        elo_pre_list.append(elo1)
        elo_pre_list.append(elo2)

        team_elo[t1] = new_elo1
        team_elo[t2] = new_elo2
    # add elo rating column to the dataframe as new column elo pre and return dataframe
    df["ELO_PRE"] = elo_pre_list[: len(df)]
    return df


# ------------------
# CURRENT SEASON MOMENTUM + BLEND

# calculate rolling averages for each teams recent performance stats over last 10 games
# capture momentum

def add_current_season_momentum(df_cur: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    print("Computing current-season momentum features ...")
    df = df_cur.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    # Store TEAM_ID separately to ensure it's preserved
    team_ids = df['TEAM_ID'].copy()

    def _roll_team(team_df: pd.DataFrame) -> pd.DataFrame:
        team_df = team_df.sort_values("GAME_DATE").copy()
        roll = (
            team_df[["FG_PCT", "FG3_PCT", "FT_PCT", "PTS", "REB", "AST", "TOV", "PLUS_MINUS", "NET_MARGIN", "WIN"]]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )
        roll.columns = [f"ROLL_{c}" for c in roll.columns]
        team_df["GAMES_PLAYED_CURR"] = (
            team_df["GAME_DATE"].rank(method="first").astype(int) - 1
        )
        team_df = pd.concat([team_df, roll], axis=1)
        team_df["RECENT_WIN_PCT"] = team_df["ROLL_WIN"]
        return team_df

    df = df.groupby("TEAM_ID", group_keys=False).apply(_roll_team)

    if 'TEAM_ID' not in df.columns:
        print("WARNING: TEAM_ID was lost, restoring it...")
        df['TEAM_ID'] = team_ids.values

    return df

# blends current season momentum stats with last season averages using Bayesian shrinkage
# reduce reliance on current-season data early in the season and rely more on last seasons stats
# as season progresses, the blend shifts toward current momentum


def blend_last_and_current(df_cur: pd.DataFrame, df_last_avg: pd.DataFrame, w_last: float = 0.15, w_cur: float = 0.85, shrink_games: int = 20) -> pd.DataFrame:
    # Merge last-season averages into current-season rows and build blended features.
    print(f"Blending last season ({w_last}) and current season ({w_cur}) stats ...")
    # merges last seaon per team avaerages into current season dataframe
    df = df_cur.merge(df_last_avg, on="TEAM_ID", how="left", suffixes=("", "_LAST"))
    # lists all the base statistics to apply the blend
    stat_bases = [
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "PTS",
        "REB",
        "AST",
        "TOV",
        "PLUS_MINUS",
        "NET_MARGIN",
    ]
    # calculates a shrinkage factor (0, if none played and 1 if enough played)
    # controls how much weight given to current v last season
    g_cur = df["GAMES_PLAYED_CURR"].fillna(0).astype(float)
    alpha = (g_cur / float(shrink_games)).clip(lower=0.0, upper=1.0)
    # iterate through each stat we want to blend
    for base in stat_bases:
        # determine the correct column names for rolling and last season versions
        cur_col = f"ROLL_{base}" if f"ROLL_{base}" in df.columns else base
        last_col = f"{base}_LAST"
        # column data for both versions
        cur_vals = df[cur_col].astype(float)
        last_vals = df[last_col].astype(float)
        # shrink current values toward last season if few games played
        shrunk_cur = alpha * cur_vals + (1.0 - alpha) * last_vals
        # combines shrunk current stats and last season stats with fixed weights
        blended = w_cur * shrunk_cur + w_last * last_vals
        df[f"BLEND_{base}"] = blended
    # return dataset
    return df


# ------------------
# GAME-LEVEL TRAINING EXAMPLES

# transforms team level game logs into game level examples for supervised learning
# processes each nba game in current season, extracts the differnece in features between home and away teams, assigns a binary label (1 win 0 loss)
# returns a feature matric (row is game column is feature difference), and a label vector

def build_game_level_examples(df_blend: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Turn team-game rows from CURRENT season into game-level rows:
    print("Building game-level training examples ...")
    # sort input dataframe by date and given id to ensure games are processed in chronological order
    df = df_blend.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    # define a list of columns to be used for training, representing team performance and elo rating
    feat_cols = [
        "BLEND_FG_PCT",
        "BLEND_FG3_PCT",
        "BLEND_FT_PCT",
        "BLEND_PTS",
        "BLEND_REB",
        "BLEND_AST",
        "BLEND_TOV",
        "BLEND_PLUS_MINUS",
        "BLEND_NET_MARGIN",
        "RECENT_WIN_PCT",
        "ELO_PRE",
    ]
    # two empty lists, games store feature dictionaries for each game and labels store outcome for each game (1 win 0 loss)
    games = []
    labels = []
    # loop through each group of rows that share same game_id
    for game_id, g in df.groupby("GAME_ID"):
        # skip games that dont have two rows
        if len(g) != 2:
            continue
        # home vs. away via MATCHUP: "XYZ vs. ABC" is home, "XYZ @ ABC" is away
        home_rows = g[g["MATCHUP"].str.contains(" vs. ", na=False)]
        away_rows = g[g["MATCHUP"].str.contains(" @ ", na=False)]
        # ensures exactly one home and one away team is found
        if len(home_rows) != 1 or len(away_rows) != 1:
            continue
        # extract series object for home and away team rows, and assign label based on home team result
        home = home_rows.iloc[0]
        away = away_rows.iloc[0]

        label = 1 if home["WL"] == "W" else 0
        # dictionary to hold feature values for this game
        feat_vals = {}
        # loop through each feature column, extracting home and away values, computing the difference and storing into dicitionary with _DIFF suffix
        for col in feat_cols:
            h_val = float(home.get(col, np.nan))
            a_val = float(away.get(col, np.nan))
            feat_vals[f"{col}_DIFF"] = h_val - a_val
        # append label and feauture dictionary to specific list
        games.append(feat_vals)
        labels.append(label)
    # convert feature dictionary to dataframe and labels into a series
    X_raw = pd.DataFrame(games)
    y_raw = pd.Series(labels, dtype=int)

    # apply mask to keep complete rows, and reset indexes of x and y so they are alligned
    mask = ~X_raw.isna().any(axis=1)
    X = X_raw[mask].reset_index(drop=True)
    y = y_raw[mask].reset_index(drop=True)

    print(f"Built {len(X)} game-level examples (after dropping NaN rows).")
    return X, y


# ------------------
# TRAINING

# trains a SVM model to predict the outcomes of nba games using last season averages, current season momentum, and elo ratings
# performs preprocessing, trains the model, evaluates it and saves to a .pkl files
def train_svm_model(last_season_label: str, current_season_label: str, w_last: float = 0.15, w_cur: float = 0.85, model_out_path: str = "models/svm_momentum_svm.pkl") -> None:
    # set up nba api session to avoid rate limits or timeouts
    configure_nba_session()

    # load data
    # fetch game logs per team for last and current seasons from nba api using predefined method
    df_last = load_season_team_logs(last_season_label)
    df_cur = load_season_team_logs(current_season_label)

    # concatenates dataset and calculates elo ratings for teams across both seasons using predefined method
    df_all = pd.concat([df_last, df_cur], ignore_index=True)
    df_all = add_elo_column(df_all)

    # keep only current season elo for training
    df_cur_with_elo = df_all[df_all["SEASON"] == current_season_label].copy()

    # calculate average stats for each team last season (method)
    last_avg = compute_last_season_team_averages(df_last)

    # compute momentum features such as rolling averages and win percentages over last n games
    df_cur_mom = add_current_season_momentum(df_cur_with_elo)
    # combine current momentum and last season stats with bayesian shrinkage (method)
    df_blend = blend_last_and_current(
        df_cur_mom,
        last_avg,
        w_last=w_last,
        w_cur=w_cur,
        shrink_games=20,
    )

    # transform team data into game level features
    X, y = build_game_level_examples(df_blend)

    # SVM model
    # save list of feature names for later use in prediction
    feature_names = list(X.columns)
    # build pipeline that scales features and SVM with balanced class weights and output probabilities
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, class_weight="balanced")),
        ]
    )
    # splits data into training and validation sets (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # train svm model on training data
    print("Training SVM model ...\n")
    clf.fit(X_train, y_train)
    # generates predictions and win probabilities for validation data
    y_pred = clf.predict(X_val)
    # y_proba = clf.predict_proba(X_val)[:, 1]
    # calculates validation accuracy
    acc = accuracy_score(y_val, y_pred)
    # prints performance on accuracy, precision, recall
    print(f"Validation accuracy: {acc:.3f}\n")
    print("Classification report:")
    print(classification_report(y_val, y_pred))

    # model bundle and prepares output path
    out_path = Path(model_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": clf,
        "feature_names": feature_names,
        "last_season": last_season_label,
        "current_season": current_season_label,
        "weights": {"last": w_last, "current": w_cur},
        "trained_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
    }
    joblib.dump(bundle, out_path)
    print(f"\nModel saved to: {out_path.resolve()}")


# main
if __name__ == "__main__":
    # define training context
    LAST_SEASON = "2024-25"
    CURRENT_SEASON = "2025-26"
    # call main function, uses two seasons data, applies blending logic and saves trained model under the pkl
    train_svm_model(
        last_season_label=LAST_SEASON,
        current_season_label=CURRENT_SEASON,
        w_last=0.15,
        w_cur=0.85,
        model_out_path=(
            "models/svm_momentum_svm.pkl"
        ),
    )
