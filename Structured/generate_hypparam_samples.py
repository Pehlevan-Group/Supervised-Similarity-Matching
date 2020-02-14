import numpy as np
import pyDOE as pd
'''
#Structured, HL=1
num_samps = 30

hpraw = pd.lhs(3, num_samps)
alphaw1 = 10.0**(-4.0 + (4.0 * hpraw[:, 0]))
tauw = hpraw[:, 1]
taul =1.0 + (hpraw[:, 2]*4.0)

alphaw2=np.multiply(alphaw1, tauw).reshape(num_samps, 1)
alphal = np.multiply(alphaw1, taul).reshape(num_samps, 1)
alphaw1 = alphaw1.reshape(num_samps, 1)

alphas = np.hstack([alphaw1, alphaw2, alphal])

alphas = np.minimum(alphas, 0.9)
np.save('lh_grid1', alphas)
'''
#Unstructured, HL=3
num_samps = 20

hpraw = pd.lhs(3, num_samps)
#hpraw = np.vstack([hpraw, np.asarray([0.5, 0.75, 1.5], dtype=np.float32), np.asarray([0.128, 0.25, 1.5], dtype=np.float32), np.asarray([0.128, 0.25, 3.0], dtype=np.float32)])
print hpraw[-1]
print hpraw[-2]
print hpraw[-3]
alphaw1 = np.vstack([(10.0**(-2.0 + (2.0 * hpraw[:, 0]))).reshape(num_samps, 1), np.asarray([0.5, 0.128, 0.128], dtype=np.float32).reshape(3, 1)])
tauw =np.vstack([hpraw[:, 1].reshape(num_samps, 1), np.asarray([0.75, 0.25, 0.25], dtype=np.float32).reshape(3, 1)])
taul = np.vstack([(1.0 + (hpraw[:, 2]*4.0)).reshape(num_samps, 1), np.asarray([1.5, 1.5, 3.0], dtype=np.float32).reshape(3, 1)])
num_samps = num_samps +3
alphaw2=np.multiply(alphaw1, tauw)
alphaw3=np.multiply(alphaw2, tauw)
alphaw4=np.multiply(alphaw3, tauw)
alphal1, alphal2, alphal3 = np.multiply(alphaw1, taul).reshape(num_samps, 1), np.multiply(alphaw2, taul).reshape(num_samps, 1), np.multiply(alphaw3, taul).reshape(num_samps, 1)
alphaw1, alphaw2, alphaw3, alphaw4 = alphaw1.reshape(num_samps, 1), alphaw2.reshape(num_samps, 1), alphaw3.reshape(num_samps, 1), alphaw4.reshape(num_samps, 1)

alphas = np.hstack([alphaw1, alphaw2, alphaw3, alphaw4, alphal1, alphal2, alphal3])
alphas = np.minimum(alphas, 0.9)
print 'Alphas'
print alphas[-1]
print alphas[-2]
print alphas[-3]
np.save('../Fast_Net3_reruns/lh_grid1', alphas)

#alphas = np.hstack([alphaw1, alphaw2, alphal])
#alphas = np.vstack([alphas, np.asarray([0.5, 0.75, 1.5], dtype=np.float32), np.asarray([], dtype=np.float32)])
