from __future__ import annotations

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import argparse

import numpy as np
import pandas as pd
import joblib
import requests as _requests
import urllib3 as _urllib3

from nba_api.stats.static import teams as static_teams

from train_svm_momentum_svm import (
    # sets up http session so we get all the data without any issues
    configure_nba_session,
    # load season team data
    load_season_team_logs,
    # get per team averages for each team
    compute_last_season_team_averages,
    # momentum from current season
    add_current_season_momentum,
    # 15% weight on last season, 85% on current
    blend_last_and_current,
    # get elo rating
    add_elo_column,
)


# look up team name and abrv
def _team_lookup():
    # api call
    teams = static_teams.get_teams()
    # create empty dict for team name and abrv
    by_id = {}
    # get team name and abr based on id and store in by_id
    for t in teams:
        by_id[int(t["id"])] = {
            "tri": t.get("abbreviation", "").upper(),
            "name": t.get("full_name", t.get("nickname", "")),
        }
    return by_id


# get the latest row of stats from the blended row based on team id up to given day
def _get_latest_team_row(df_blend: pd.DataFrame, team_id: int, cutoff: pd.Timestamp):
    # set d to the rows of the given team
    d = df_blend[df_blend["TEAM_ID"] == team_id].copy()
    if d.empty:
        return None
    # filter out games that occur after the prediction
    d = d[d["GAME_DATE"] <= cutoff]
    if d.empty:
        return None
    # make sure games are in correct order
    d = d.sort_values("GAME_DATE")
    # get most recent row
    return d.iloc[-1]


# main prediction logic
def predict_for_date(
    target_iso: str,
    model_path: str = "models/svm_momentum_svm.pkl",
    quiet: bool = False,
    save_csv: bool = True,
) -> pd.DataFrame:
    # load model
    bundle = joblib.load(model_path)
    # get the model
    model = bundle["model"]
    # get all the features (stats)
    feature_names = bundle["feature_names"]
    # get last and current seaons
    last_season = bundle["last_season"]
    current_season = bundle["current_season"]
    # weigh model based on split for last and current seaon
    w_last = bundle["weights"]["last"]
    w_cur = bundle["weights"]["current"]

    # confrim stuff is being loaded properly
    if not quiet:
        print(f"Loaded model from {model_path}")
        print(f"Last season: {last_season}, current season: {current_season}")
        print(f"Weights → last: {w_last}, current: {w_cur}")

    # configure nba_api session with retry logic, etc (refer to function in training)
    configure_nba_session()

    # fix date format
    try:
        target_dt = datetime.strptime(target_iso, "%Y-%m-%d").replace(
            tzinfo=ZoneInfo("America/New_York")
        )
    except ValueError:
        if not quiet:
            print(f"Invalid date '{target_iso}', expected YYYY-MM-DD.")
        return pd.DataFrame([])

    target_mmddyyyy = target_dt.strftime("%m/%d/%Y")
    cutoff_dt = pd.to_datetime(target_iso)

    if not quiet:
        print(f"\nBuilding predictions for games on {target_iso} ...")

    # Use CDN instead of stats.nba.com (blocked on AWS)
    import requests as _requests
    import urllib3 as _urllib3
    _urllib3.disable_warnings()

    is_today = target_iso == datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    if is_today:
        cdn_url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
    else:
        cdn_url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"

    cdn_resp = _requests.get(cdn_url, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com"
    }, timeout=15, verify=False)
    cdn_data = cdn_resp.json()

    if is_today:
        cdn_games = cdn_data.get("scoreboard", {}).get("games", [])
    else:
        game_dates = cdn_data.get("leagueSchedule", {}).get("gameDates", [])
        cdn_games = []
        for gd in game_dates:
            raw_date = gd.get("gameDate", "")
            try:
                gd_str = datetime.strptime(raw_date, "%m/%d/%Y %H:%M:%S").strftime("%Y-%m-%d")
            except Exception:
                continue
            if gd_str == target_iso:
                cdn_games = gd.get("games", [])
                break

    if not cdn_games:
        if not quiet:
            print("No games found for this date.")
        return pd.DataFrame([])

    gh_rows = []
    for g in cdn_games:
        ht = g.get("homeTeam", {})
        at = g.get("awayTeam", {})
        gh_rows.append({
            "GAME_ID": g.get("gameId", ""),
            "HOME_TEAM_ID": ht.get("teamId"),
            "VISITOR_TEAM_ID": at.get("teamId"),
        })
    gh = pd.DataFrame(gh_rows)

    # get teams
    teams_by_id = _team_lookup()

    # load both seasons data
    df_last = load_season_team_logs(last_season)
    df_cur = load_season_team_logs(current_season)

    # combine both seasons into one dataframe
    df_all = pd.concat([df_last, df_cur], ignore_index=True)
    # add elo column
    df_all = add_elo_column(df_all)
    # get current season data and remove last seasons while keeping elo for momentum purpose
    df_cur_with_elo = df_all[df_all["SEASON"] == current_season].copy()
    last_avg = compute_last_season_team_averages(df_last)
    # add momentum factor (last 10 games)
    df_cur_mom = add_current_season_momentum(df_cur_with_elo)
    # combine last seasons stats with current momentum and elo
    df_blend = blend_last_and_current(
        df_cur_mom,
        last_avg,
        w_last=w_last,
        w_cur=w_cur,
        shrink_games=20,
    )
    # add date to dataframe
    df_blend["GAME_DATE"] = pd.to_datetime(df_blend["GAME_DATE"])

    # storage for feature vectors (ex. point diff -3, etc)
    rows = []
    # storage for labels (team name, abr)
    meta = []

    # get game header row for each of todays games
    for _, r in gh.iterrows():
        game_id = r["GAME_ID"]
        home_id = int(r["HOME_TEAM_ID"] or 0)
        away_id = int(r["VISITOR_TEAM_ID"] or 0)
        if not home_id or not away_id:
            continue

        home_row = _get_latest_team_row(df_blend, home_id, cutoff_dt)
        away_row = _get_latest_team_row(df_blend, away_id, cutoff_dt)
        if home_row is None or away_row is None:
            if not quiet:
                print(f"Skipping {game_id} (missing blended stats for one team).")
            continue

        feat_vals = {}
        base_stats = [
            "FG_PCT", "FG3_PCT", "FT_PCT", "PTS", "REB",
            "AST", "TOV", "PLUS_MINUS", "NET_MARGIN",
        ]

        for base in base_stats:
            h_val = float(home_row.get(f"BLEND_{base}", np.nan))
            a_val = float(away_row.get(f"BLEND_{base}", np.nan))
            feat_vals[f"BLEND_{base}_DIFF"] = h_val - a_val

        h_recent = float(home_row.get("RECENT_WIN_PCT", np.nan))
        a_recent = float(away_row.get("RECENT_WIN_PCT", np.nan))
        feat_vals["RECENT_WIN_PCT_DIFF"] = h_recent - a_recent

        h_elo = float(home_row.get("ELO_PRE", np.nan))
        a_elo = float(away_row.get("ELO_PRE", np.nan))
        feat_vals["ELO_PRE_DIFF"] = h_elo - a_elo

        rows.append(feat_vals)

        h_meta = teams_by_id.get(home_id, {"name": f"Team {home_id}", "tri": ""})
        a_meta = teams_by_id.get(away_id, {"name": f"Team {away_id}", "tri": ""})
        meta.append({
            "game_id": game_id,
            "home_id": home_id,
            "away_id": away_id,
            "home_name": h_meta["name"],
            "away_name": a_meta["name"],
            "home_tri": h_meta["tri"],
            "away_tri": a_meta["tri"],
        })

    if not rows:
        if not quiet:
            print("No games with enough data to predict.")
        return pd.DataFrame([])

    X_date = pd.DataFrame(rows)
    X_date = X_date.reindex(columns=feature_names, fill_value=0.0)
    X_date = X_date.fillna(0.0)

    probs = model.predict_proba(X_date)[:, 1]

    if not quiet:
        print("\n=== A Model Prediction ===\n")
        for info, p in zip(meta, probs):
            home = f"{info['home_name']} ({info['home_tri']})"
            away = f"{info['away_name']} ({info['away_tri']})"
            print(f"{target_iso} — {home} vs {away} → P(home win) = {p:.3f}")

    out_rows = []
    for info, p in zip(meta, probs):
        out_rows.append({
            "date": target_iso,
            "game_id": info["game_id"],
            "home_id": info["home_id"],
            "away_id": info["away_id"],
            "home_name": info["home_name"],
            "away_name": info["away_name"],
            "home_tri": info["home_tri"],
            "away_tri": info["away_tri"],
            "p_home_win": float(p),
        })

    out_df = pd.DataFrame(out_rows)

    if save_csv:
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        csv_path = out_dir / f"a_svm_predictions_{target_iso}.csv"
        out_df.to_csv(csv_path, index=False)
        if not quiet:
            print(f"\nPredictions saved to: {csv_path.resolve()}")

    return out_df


# predict for today
def predict_today(model_path: str = "models/svm_momentum_svm.pkl", date_str: str | None = None) -> None:
    if date_str:
        target_iso = date_str
    else:
        now_et = datetime.now(ZoneInfo("America/New_York"))
        target_iso = now_et.strftime("%Y-%m-%d")

    # preserve old behavior: prints + saves CSV
    predict_for_date(target_iso, model_path=model_path, quiet=False, save_csv=True)


# helper functions for Flask API calls
def get_predictions_for_date(
    date_str: str,
    model_path: str = "models/svm_momentum_svm.pkl",
) -> pd.DataFrame:
    # return df but do not print or write csv
    return predict_for_date(date_str, model_path=model_path, quiet=True, save_csv=False)

def get_predictions_with_features_for_date(
    date_str: str,
    model_path: str = "models/svm_momentum_svm.pkl",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      preds_df: same schema as get_predictions_for_date (date, game_id, teams, p_home_win, etc.)
      X_date: feature-differential matrix used for those exact predictions (columns match model feature_names)

    Notes:
      - This does NOT write CSV.
      - This does NOT print.
      - This does NOT change how predictions are computed.
    """
    # load model bundle
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    last_season = bundle["last_season"]
    current_season = bundle["current_season"]
    w_last = bundle["weights"]["last"]
    w_cur = bundle["weights"]["current"]

    configure_nba_session()

    # parse date
    try:
        target_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
            tzinfo=ZoneInfo("America/New_York")
        )
    except ValueError:
        return pd.DataFrame([]), pd.DataFrame([])

    target_mmddyyyy = target_dt.strftime("%m/%d/%Y")
    cutoff_dt = pd.to_datetime(date_str)

    # Use CDN instead of stats.nba.com (blocked on AWS)
    import requests as _requests
    import urllib3 as _urllib3
    _urllib3.disable_warnings()

    is_today = date_str == datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    if is_today:
        cdn_url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
    else:
        cdn_url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"

    cdn_resp = _requests.get(cdn_url, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com"
    }, timeout=15, verify=False)
    cdn_data = cdn_resp.json()

    if is_today:
        cdn_games = cdn_data.get("scoreboard", {}).get("games", [])
    else:
        game_dates = cdn_data.get("leagueSchedule", {}).get("gameDates", [])
        cdn_games = []
        for gd in game_dates:
            raw_date = gd.get("gameDate", "")
            try:
                gd_str = datetime.strptime(raw_date, "%m/%d/%Y %H:%M:%S").strftime("%Y-%m-%d")
            except Exception:
                continue
            if gd_str == date_str:
                cdn_games = gd.get("games", [])
                break

    if not cdn_games:
        return pd.DataFrame([]), pd.DataFrame([])

    gh_rows = []
    for g in cdn_games:
        ht = g.get("homeTeam", {})
        at = g.get("awayTeam", {})
        gh_rows.append({
            "GAME_ID": g.get("gameId", ""),
            "HOME_TEAM_ID": ht.get("teamId"),
            "VISITOR_TEAM_ID": at.get("teamId"),
        })
    gh = pd.DataFrame(gh_rows)

    teams_by_id = _team_lookup()

    # load logs + build blended features
    df_last = load_season_team_logs(last_season)
    df_cur = load_season_team_logs(current_season)

    df_all = pd.concat([df_last, df_cur], ignore_index=True)
    df_all = add_elo_column(df_all)

    df_cur_with_elo = df_all[df_all["SEASON"] == current_season].copy()
    last_avg = compute_last_season_team_averages(df_last)
    df_cur_mom = add_current_season_momentum(df_cur_with_elo)

    df_blend = blend_last_and_current(
        df_cur_mom,
        last_avg,
        w_last=w_last,
        w_cur=w_cur,
        shrink_games=20,
    )
    df_blend["GAME_DATE"] = pd.to_datetime(df_blend["GAME_DATE"])

    rows = []
    meta = []

    for _, r in gh.iterrows():
        game_id = r["GAME_ID"]
        home_id = int(r["HOME_TEAM_ID"] or 0)
        away_id = int(r["VISITOR_TEAM_ID"] or 0)
        if not home_id or not away_id:
            continue

        home_row = _get_latest_team_row(df_blend, home_id, cutoff_dt)
        away_row = _get_latest_team_row(df_blend, away_id, cutoff_dt)
        if home_row is None or away_row is None:
            continue

        feat_vals = {}
        base_stats = [
            "FG_PCT", "FG3_PCT", "FT_PCT", "PTS", "REB",
            "AST", "TOV", "PLUS_MINUS", "NET_MARGIN",
        ]

        for base in base_stats:
            h_val = float(home_row.get(f"BLEND_{base}", np.nan))
            a_val = float(away_row.get(f"BLEND_{base}", np.nan))
            feat_vals[f"BLEND_{base}_DIFF"] = h_val - a_val

        h_recent = float(home_row.get("RECENT_WIN_PCT", np.nan))
        a_recent = float(away_row.get("RECENT_WIN_PCT", np.nan))
        feat_vals["RECENT_WIN_PCT_DIFF"] = h_recent - a_recent

        h_elo = float(home_row.get("ELO_PRE", np.nan))
        a_elo = float(away_row.get("ELO_PRE", np.nan))
        feat_vals["ELO_PRE_DIFF"] = h_elo - a_elo

        rows.append(feat_vals)

        h_meta = teams_by_id.get(home_id, {"name": f"Team {home_id}", "tri": ""})
        a_meta = teams_by_id.get(away_id, {"name": f"Team {away_id}", "tri": ""})
        meta.append({
            "game_id": game_id,
            "home_id": home_id,
            "away_id": away_id,
            "home_name": h_meta["name"],
            "away_name": a_meta["name"],
            "home_tri": h_meta["tri"],
            "away_tri": a_meta["tri"],
        })

    if not rows:
        return pd.DataFrame([]), pd.DataFrame([])

    X_date = pd.DataFrame(rows)
    X_date = X_date.reindex(columns=feature_names, fill_value=0.0).fillna(0.0)

    probs = model.predict_proba(X_date)[:, 1]

    out_rows = []
    for info, p in zip(meta, probs):
        out_rows.append({
            "date": date_str,
            "game_id": info["game_id"],
            "home_id": info["home_id"],
            "away_id": info["away_id"],
            "home_name": info["home_name"],
            "away_name": info["away_name"],
            "home_tri": info["home_tri"],
            "away_tri": info["away_tri"],
            "p_home_win": float(p),
        })

    preds_df = pd.DataFrame(out_rows)
    return preds_df, X_date

def get_prediction_for_game(
    date_str: str,
    game_id: str,
    model_path: str = "models/svm_momentum_svm.pkl",
) -> dict | None:
    # pull all predictions for date, filter by game_id
    df = get_predictions_for_date(date_str, model_path=model_path)
    if df is None or df.empty:
        return None
    hit = df[df["game_id"].astype(str) == str(game_id)]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()

def get_prediction_with_features_for_game(
    date_str: str,
    game_id: str,
    model_path: str = "models/svm_momentum_svm.pkl",
) -> tuple[dict | None, dict | None]:
    preds_df, X_date = get_predictions_with_features_for_date(date_str, model_path=model_path)
    if preds_df is None or preds_df.empty:
        return None, None
    preds_df = preds_df.reset_index(drop=True)
    hit_idx = preds_df.index[preds_df["game_id"].astype(str) == str(game_id)]
    if len(hit_idx) == 0:
        return None, None
    i = int(hit_idx[0])
    return preds_df.iloc[i].to_dict(), X_date.iloc[i].to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        # using --date followed by YYYY-MM-DD we can predict for any given day
        "--date",
        type=str,
        help="put date in YYYY-MM-DD format",
    )
    args = parser.parse_args()

    if args.date:
        predict_today(date_str=args.date)
    else:
        predict_today()