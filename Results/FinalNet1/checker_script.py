import sys
import numpy as np

fstate = np.load(sys.argv[1])
print "Hparams, ", fstate[-2]
tr, val = fstate[-1]['training error'], fstate[-1]['validation error']
print 'At 25'
print "TrErr", tr[24]
print "ValErr",  val[24]

if len(tr)>=40:
	print 'At 40'
	print "TrErr", tr[39]
	print "ValErr",  val[39]

print 'Final'
print "TrErr", len(tr), tr[-1]
print "ValErr", len(val), val[-1]

