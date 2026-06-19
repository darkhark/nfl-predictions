# tests/data/madden/test_roles.py
import unittest
import pandas as pd
from src.data.madden import roles


class TestAssignRole(unittest.TestCase):
    def test_olb_high_passrush_is_edge(self):
        # 3-4 edge: LOLB with power+finesse >= 130
        self.assertEqual(roles.assign_role('LOLB', 250, 88, 90), 'edge')

    def test_olb_low_passrush_is_off_ball(self):
        # 4-3 weakside backer: OLB that can't rush
        self.assertEqual(roles.assign_role('ROLB', 240, 40, 45), 'off_ball_lb')

    def test_light_de_is_edge_heavy_de_is_interior(self):
        self.assertEqual(roles.assign_role('LE', 270, 80, 85), 'edge')
        self.assertEqual(roles.assign_role('RE', 295, 70, 60), 'interior_dl')

    def test_dt_and_mlb_unambiguous(self):
        self.assertEqual(roles.assign_role('DT', 310, 70, 40), 'interior_dl')
        self.assertEqual(roles.assign_role('MLB', 240, 30, 35), 'off_ball_lb')

    def test_modern_labels_map_directly(self):
        self.assertEqual(roles.assign_role('LEDGE', 250, 0, 0), 'edge')
        self.assertEqual(roles.assign_role('MIKE', 240, 0, 0), 'off_ball_lb')
        self.assertEqual(roles.assign_role('SAM', 240, 0, 0), 'off_ball_lb')

    def test_offense_and_dbs(self):
        self.assertEqual(roles.assign_role('LT', 320, 0, 0), 'exterior_ol')
        self.assertEqual(roles.assign_role('C', 300, 0, 0), 'interior_ol')
        self.assertEqual(roles.assign_role('CB', 190, 0, 0), 'cornerback')
        self.assertEqual(roles.assign_role('FS', 200, 0, 0), 'safety')
        self.assertEqual(roles.assign_role('QB', 220, 0, 0), 'qb')


class TestAssignSide(unittest.TestCase):
    def test_side_from_label(self):
        self.assertEqual(roles.assign_side('LE'), 'left')
        self.assertEqual(roles.assign_side('REDGE'), 'right')
        self.assertEqual(roles.assign_side('DT'), 'none')


class TestClassifyRoles(unittest.TestCase):
    def test_adds_role_and_side_columns(self):
        df = pd.DataFrame([{'position': 'LOLB', 'weight': 250,
                            'power_moves': 88, 'finesse_moves': 90}])
        out = roles.classify_roles(df)
        self.assertEqual(out.iloc[0]['role'], 'edge')
        self.assertEqual(out.iloc[0]['side'], 'left')
