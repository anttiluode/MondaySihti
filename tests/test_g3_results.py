import json
import unittest
from pathlib import Path

class FrozenG3ResultTest(unittest.TestCase):
    def test_frozen_seeds_and_split_sizes(self):
        r=json.loads((Path(__file__).resolve().parents[1]/'data/nonlinear_results.json').read_text())
        self.assertEqual(r['schema'],'mondaysihti.nonlinear.v1')
        self.assertEqual(r['seeds'],[101,211,307,419,523])
        self.assertEqual(r['data_sizes'],[96,48,64])
        self.assertEqual(len(r['by_seed']),5)
        self.assertTrue(r['gates']['G3_1_exact_residue_inverse'])
        self.assertFalse(r['gates']['G3_2_residue_5pct_better_than_matched'])
        self.assertFalse(r['gates']['G3_3_both_histories_5pct_better_than_snapshot'])
    def test_missing_history_is_not_an_internal_label(self):
        r=json.loads((Path(__file__).resolve().parents[1]/'data/nonlinear_results.json').read_text())
        for row in r['by_seed']:
            ex=row['example']
            self.assertEqual(len(ex['pre_soma_observed']),8)
            self.assertEqual(len(ex['response_actual']),4)
            self.assertEqual(len(ex['predictions']['residue_three']),4)
            self.assertNotIn('hidden_ports',ex)

if __name__=='__main__':unittest.main()
