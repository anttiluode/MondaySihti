import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.bridge import (image_plant,cable_plant,observation,octave_difference,
                                synthetic_states,benchmark_one,TIMES)

class BridgeTest(unittest.TestCase):
    def test_image_operator_is_stochastic(self):
        p,c,_=image_plant()
        self.assertEqual(p.shape,(64,64))
        self.assertTrue(np.allclose(p.sum(axis=1),1))
        self.assertTrue(np.min(p)>=0)
    def test_cable_eigenvalues_stable(self):
        p,c=cable_plant()
        self.assertEqual(c.shape,(1,7))
        self.assertLess(max(abs(np.linalg.eigvals(p))),1)
    def test_dyadic_residue_is_exact(self):
        d=octave_difference()
        rng=np.random.default_rng(2)
        y=rng.normal(size=(7,4))
        z=d@y
        self.assertLess(np.max(np.abs(z.sum(0)-y[0])),1e-14)
        self.assertLess(np.max(np.abs(np.linalg.solve(d,z)-y)),1e-13)
    def test_rank_preserved_both_domains(self):
        plants=[image_plant()[:2],cable_plant()]
        for p,c in plants:
            o=observation(p,c)
            d=np.kron(octave_difference(),np.eye(len(c)))
            self.assertEqual(np.linalg.matrix_rank(o),np.linalg.matrix_rank(d@o))
            self.assertLess(np.max(np.abs(np.linalg.solve(d,d@o)-o)),1e-12)
    def test_reconstructed_fields(self):
        for key, (p,c) in [('image',image_plant()[:2]),('cable',cable_plant())]:
            rng=np.random.default_rng(42)
            x=synthetic_states(key,p,rng,1)[0]
            states=np.stack([np.linalg.matrix_power(p,t)@x for t in TIMES])
            bands=octave_difference()@states
            self.assertLess(np.max(np.abs(bands.sum(0)-states[0])),1e-13)
    def test_sensor_not_state(self):
        p,c,_=image_plant()
        self.assertLess(np.linalg.matrix_rank(observation(p,c)),len(p))
    def test_repeatability(self):
        p,c=cable_plant()
        a=benchmark_one('cable',p,c,11)
        b=benchmark_one('cable',p,c,11)
        self.assertEqual(a['methods'],b['methods'])
    def test_holdout_separated(self):
        from experiments.bridge import N_TRAIN,N_VAL,N_TEST
        self.assertEqual((N_TRAIN,N_VAL,N_TEST),(400,140,180))

if __name__=='__main__':unittest.main()
