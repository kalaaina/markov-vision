import gibbs_sampler
import numpy as np

H,W = 32,32
fake = np.random.randint(0,3,(H,W))
img = np.zeros((H,W,3), dtype=np.uint8)
final, hist = gibbs_sampler.run_gibbs(fake, img, n_iter=2)
print('final.shape=', final.shape, 'history_len=', len(hist))
print('OK')
