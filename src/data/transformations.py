import pandas as pd

# If a team moved cities, the name in current data releases is the most recent name.
# Older sources (schedules, play-by-play) use the era abbreviation; map to the current
# one so frames merge cleanly on team.
TEAM_ABBR_MAPPINGS = {
    'STL': 'LA',
    'SD': 'LAC',
    'OAK': 'LV'
}

TEAM_COL = 'team'
OPPONENT_TEAM_COL = 'opp_team'
SEASON_COL = 'season'
WEEK_COL = 'week'
TEAM_GAME_COUNT_COL = 'team_game_count'
OPP_GAME_COUNT_COL = 'opp_game_count'


def add_rank_and_rank_change_columns(df, off_cols, def_cols):
    """
    Add cross-sectional rank and week-over-week rank-change columns for the given offense
    and defense columns, computed within each (season, week).

    Offense columns are ranked descending, so rank 1 is the highest value (best offense).
    Defense columns are ranked ascending, so rank 1 is the fewest allowed (best defense).
    Ties share the better rank (competition ranking), matching how league standings are
    reported. NaN values receive NaN ranks; remaining teams are ranked among themselves.

    Rank-change is the per-team change in that rank from the previous game: offense ranks
    are diffed within (team, season) and defense ranks within (opp_team, season), because
    def_opp_* columns describe the opponent's defense. Concretely, the defense rank change
    on a row is the change in the defending team's (i.e. opp_team's) rank from its own
    previous game. The first game of each group has a rank-change of 0.

    :param df: a team-week frame with team/opp_team/season/week/game-count columns
    :param off_cols: offense columns to rank (higher is better)
    :param def_cols: defense columns to rank (lower is better)
    :return: the frame with *_rank and *_rank_change columns appended
    """
    off_ranks = df.groupby([SEASON_COL, WEEK_COL])[off_cols].rank(ascending=False, method='min').add_suffix('_rank')
    def_ranks = df.groupby([SEASON_COL, WEEK_COL])[def_cols].rank(ascending=True, method='min').add_suffix('_rank')
    df = pd.concat([df, off_ranks, def_ranks], axis=1)

    off_rank_changes = rank_change_frame(df, list(off_ranks.columns), [TEAM_COL, SEASON_COL], TEAM_GAME_COUNT_COL)
    def_rank_changes = rank_change_frame(df, list(def_ranks.columns), [OPPONENT_TEAM_COL, SEASON_COL], OPP_GAME_COUNT_COL)
    df = pd.concat([df, off_rank_changes, def_rank_changes], axis=1)

    df.sort_values(by=[TEAM_COL, SEASON_COL, TEAM_GAME_COUNT_COL], inplace=True)
    return df


def rank_change_frame(df, rank_cols, groupby_columns, game_count_col):
    """
    Build a *_rank_change frame for each rank column, computed as the change from the
    entity's previous game. The entity is whatever groupby_columns identifies (the team for
    offense ranks, the opponent for defense ranks). The first game of each group is filled
    with 0. The returned frame is indexed like df so it can be concatenated back on.
    """
    ordered = df.sort_values(by=groupby_columns + [game_count_col])
    changes = ordered.groupby(groupby_columns)[rank_cols].diff().fillna(0)
    return changes.add_suffix('_change')
