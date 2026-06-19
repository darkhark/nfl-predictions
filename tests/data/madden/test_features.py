import unittest
import pandas as pd
from src.data.madden import features


class TestBuildTeamWeekOveralls(unittest.TestCase):
    def _starters(self):
        # one team-week: QB, two tackles, three WRs, a left edge
        return pd.DataFrame([
            {'season': 2023, 'week': 1, 'team': 'KC', 'gsis_id': g, 'position': p}
            for g, p in [('QB1', 'QB'), ('T1', 'T'), ('T2', 'T'),
                         ('W1', 'WR'), ('W2', 'WR'), ('W3', 'WR'), ('E1', 'OLB')]
        ])

    def _players(self):
        return pd.DataFrame([
            {'gsis_id': 'QB1', 'overall': 96, 'position': 'QB', 'role': 'qb', 'side': 'none'},
            {'gsis_id': 'T1', 'overall': 90, 'position': 'LT', 'role': 'exterior_ol', 'side': 'left'},
            {'gsis_id': 'T2', 'overall': 78, 'position': 'RT', 'role': 'exterior_ol', 'side': 'right'},
            {'gsis_id': 'W1', 'overall': 99, 'position': 'WR', 'role': 'receiver', 'side': 'none'},
            {'gsis_id': 'W2', 'overall': 84, 'position': 'WR', 'role': 'receiver', 'side': 'none'},
            {'gsis_id': 'W3', 'overall': 72, 'position': 'WR', 'role': 'receiver', 'side': 'none'},
            {'gsis_id': 'E1', 'overall': 88, 'position': 'LOLB', 'role': 'edge', 'side': 'left'},
        ])

    def test_granular_slots_and_groups(self):
        out = features.build_team_week_overalls(self._starters(), self._players())
        row = out.iloc[0]
        self.assertEqual(row['madden_qb_ovr'], 96)
        self.assertEqual(row['madden_lt_ovr'], 90)    # side from Madden position
        self.assertEqual(row['madden_rt_ovr'], 78)
        self.assertEqual(row['madden_wr1_ovr'], 99)   # ranked by overall desc
        self.assertEqual(row['madden_wr2_ovr'], 84)
        self.assertEqual(row['madden_wr3_ovr'], 72)
        self.assertEqual(row['madden_edge_left_ovr'], 88)
        self.assertEqual(row['madden_receivers_ovr'], (99 + 84 + 72) / 3)
        self.assertEqual(row['madden_exterior_ol_ovr'], (90 + 78) / 2)


if __name__ == '__main__':
    unittest.main()
