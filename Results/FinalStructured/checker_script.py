import sys
import numpy as np

fstate = np.load(sys.argv[1])
print "Hparams, ", fstate[-2]
tr, val = fstate[-1]['training error'], fstate[-1]['validation error']

if len(tr)>=25:
	print 'At 25'
	print "TrErr", tr[20]
	print "ValErr",  val[2]

print 'Final'
print "TrErr", len(tr), tr[-1]
print "ValErr", len(val), val[-1]

