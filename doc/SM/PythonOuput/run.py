import sys

from SM import RGEsolver

##############################################
# First, create an instance of the RGEsolver #
##############################################

rge = RGEsolver('rge', tmin=0, tmax=20, initialScale=0)


##########################################################
# We fix the running scheme and initial conditions below #
##########################################################

# Running scheme :

rge.loops = {'GaugeCouplings': 2,
             'Yukawas': 2,
             'QuarticTerms': 2,
             'ScalarMasses': 2,
             'Vevs': 2}

# Gauge Couplings

rge.g1.initialValue = 0
rge.g2.initialValue = 0
rge.g3.initialValue = 0

# Yukawa Couplings

rge.yt.initialValue = 0
rge.yb.initialValue = 0
rge.ytau.initialValue = 0

# Quartic Couplings

rge.lambda_.initialValue = 0

# Scalar Mass Couplings

rge.mu.initialValue = 0

# Vacuum-expectation Values

rge.v.initialValue = 0


############################
# Solve the system of RGEs #
############################

rge.solve(step = .05)

# Another way to call rge.solve() :
# rge.solve(Npoints = 500)

####################
# Plot the results #
####################

rge.plot(subPlots=True, printLoopLevel=True)

