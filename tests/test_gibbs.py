import unittest
import numpy as np
import gibbs_sampler

class TestGibbsSampler(unittest.TestCase):
    def test_run_gibbs_shape_and_history(self):
        """Test that run_gibbs returns the correct shape and energy history length."""
        H, W = 32, 32
        # Setup fake labels (3 classes) and dummy source image
        labels = np.random.randint(0, 3, (H, W))
        img = np.zeros((H, W, 3), dtype=np.uint8)
        
        n_iter = 2
        final_labels, energy_history = gibbs_sampler.run_gibbs(
            labels=labels,
            image=img,
            beta=1.5,
            temperature=1.0,
            n_iter=n_iter
        )
        
        # Verify shape remains the same
        self.assertEqual(final_labels.shape, (H, W))
        # Verify history matches expectation (initial energy + energy after each iteration)
        self.assertEqual(len(energy_history), n_iter + 1)
        # Verify labels remain within valid range
        self.assertTrue(np.all(final_labels >= 0))
        self.assertTrue(np.all(final_labels < 3))

if __name__ == '__main__':
    unittest.main()
