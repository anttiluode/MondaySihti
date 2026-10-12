"""G3 checks: electrical boundary, target independence, no hidden causes leaked."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.nonlinear import (ReferenceCable,program,residue8,restore8,
    run_seed,TRAIN_PAIRS,VAL_PAIRS,TEST_PAIRS,PRE_IDX,POST_IDX,
    SITES,features,PRE_TIMES)

class NonlinearGateTest(unittest.TestCase):
    def test_hidden_port_pair_families_disjoint(self):
        a,b,c=map(set,[TRAIN_PAIRS,VAL_PAIRS,TEST_PAIRS])
        self.assertFalse(a&b);self.assertFalse(a&c);self.assertFalse(b&c)
        self.assertEqual(len(a|b|c),16)
    def test_sensors_before_probe(self):
        self.assertEqual(len(PRE_IDX),8)
        self.assertEqual(len(POST_IDX),4)
        self.assertLess(max(PRE_IDX),min(POST_IDX))
        self.assertLess(max(PRE_TIMES),18.7)
    def test_residue_is_full_rank_coordinate_change(self):
        rng=np.random.default_rng(21)
        x=rng.normal(size=(19,8))
        self.assertLess(np.max(np.abs(restore8(residue8(x))-x)),1e-13)
        self.assertEqual(np.linalg.matrix_rank(np.vstack([residue8(np.eye(8)),np.zeros((2,8))])),8)
    def test_quadratic_capacity_equal_three_scalar_methods(self):
        self.assertEqual(features(np.zeros((3,3))).shape,(3,9))
        self.assertEqual(features(np.zeros((3,1))).shape,(3,2))
    def test_zero_conductance_stays_at_rest(self):
        model=ReferenceCable();n=len(SITES)
        res=model.run(np.zeros((n,7)),np.zeros((n,7)))
        self.assertTrue(res.all_steps_converged)
        self.assertEqual(np.max(np.abs(res.soma_depolarization_mV)),0.)
    def test_probe_protocol_keeps_history_identical(self):
        rng=np.random.default_rng(1)
        ga,gn,pga,pgn=program(rng,(0,3))
        self.assertTrue(np.array_equal(ga[:,:124],pga[:,:124]))
        self.assertTrue(np.array_equal(gn[:,:124],pgn[:,:124]))
        self.assertEqual(np.sum(np.abs(ga-pga)>0),np.count_nonzero(ga-pga))
        self.assertGreater(np.max(pgn-gn),0)
    def test_pilot_repeats(self):
        runner=ReferenceCable()
        a=run_seed(1,runner,ntrain=6,nval=5,ntest=5)
        b=run_seed(1,runner,ntrain=6,nval=5,ntest=5)
        self.assertEqual(a['methods'],b['methods'])
        self.assertEqual(len(a['example']['response_actual']),4)
        self.assertLess(a['diagnostics']['residue_roundtrip_max'],1e-12)

if __name__=='__main__':unittest.main()
