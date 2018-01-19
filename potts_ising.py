#!/usr/bin/env python 

import numpy as np
import random, sys, os
#from scipy import sparse
import math
from math import exp, sin, cos, pi
#from outclass import TerminalController
#from optparse import OptionParser
from commands import getstatusoutput
from time import sleep, time
#import urllib2

# Coupling constant * 1/Temperature
#J = 1.0  # positive = ferromagnetic

# Inverse temperature ( 1 / k_B T)
beta = 1.0 / 1.0


# System size, periodic B.C. by default
L = 16

# Create, but not initialize, L x L array of double-precision floats
spins = None

# Potts model
q = 0


def initialize_lattice():
	global spins

	spins = np.empty([L,L], dtype=int)

	# Ising model
	choices = [-1, 1]
	
	while True:
		for i in range(L):
			for j in range(L):
				spins[i,j] = choices[random.random() > 0.5]
		if spins.sum() < L:
			break		


	en = 0.0
	for i in range(L):
		for j in range(L):
			# Go over half the bonds of each site, thus going over all lattice bonds ONCE
			#print '%d \t %d \t %d \t %d \n' % (i-1 % L, i+1 % L, j-1 % L, j+1 % L)
			en += spins[i,j] * (spins[ (i+1) % L,j] + spins[i, (j + 1) % L]) # Calculate bond energy
	return [spins.sum(), en]


def initialize_potts():
	global spins

	spins = np.empty([L,L], dtype=int)

	# Ising model
	choices = range(1, q+1)
	
	for i in range(L):
		for j in range(L):
			spins[i,j] = random.choice( choices )

	en = 0.0
	for i in range(L):
		for j in range(L):
			# Go over half the bonds of each site, thus going over all lattice bonds ONCE
			#print '%d \t %d \t %d \t %d \n' % (i-1 % L, i+1 % L, j-1 % L, j+1 % L)
			en += (spins[i,j] == spins[ (i+1) % L,j])
			en += (spins[i,j] == spins[i, (j + 1) % L])
			en *= -1.0
	return [spins.sum(), en]



def energy(_site, val):
	''' Calculates the energy on the bonds of a particular site ''' 
	assert (type(_site) == list) or (type(_site) == tuple)
	
	[i,j] = _site
	H = spins[ (i+1) % L, j] + spins[i, (j+1) % L] + spins[ (i-1) % L,j] + spins[i, (j-1) % L] 
	H *= val

	# Ferromagnetic Ising Model
	H *= -1.0
	return H

def pottsenergy(_site, val):
	''' Calculates the energy on the bonds of a particular site in the Potts model''' 
	[i,j] = _site
	H = 0
	H = int((spins[ (i+1) % L, j] == val ))
	H += (spins[ (i-1) % L, j] == val )
	H += (spins[i, (j+1) % L] == val )
	H += (spins[i, (j-1) % L] == val )
	H *= -1.0
	return H

def step():
	''' Attempts to flip a single random spin. Returns [ magnetization change, energy change ] '''
	global spins
	
	#[i,j] = [random.choice(range(L)), random.choice(range(L))]
	[i,j] = [ int(math.ceil( random.random() * L**5)) % L,  int(math.ceil( random.random() * L**5)) % L ] 
	current = spins[i,j]
	en = energy([i,j], current)
	# It suffices to calculate the sum of S_j's that are NNs to i
	# and then the two possible energies are +/- beta * J * this_sum
	if (en > 0) or exp(2 * en * beta) > random.random():  #if en > 0, then  -en < 0, meaning the opposite configuration has lower energy
		spins[i,j] *= -1
		return [-2 * current, -2 * en]
	
	# No flip
	return [0.0, 0.0]

def potts_step():
	global spins

	[i,j] = [ int(math.ceil( random.random() * L**5)) % L,  int(math.ceil( random.random() * L**5)) % L ] 
	current = spins[i,j]

	choices = filter(lambda x: x != current, range(1,q+1))
	new_spin = random.choice( choices )

	en1 = pottsenergy([i,j], current)
	en2 = pottsenergy([i,j], new_spin)
	en = en1 - en2

	if (en > 0) or exp( en * beta) > random.random():
		spins[i,j] = new_spin
		return [new_spin - current, -en]
	
	# No state change
	return [0.0, 0.0]	
	

def total_energy():
	en = 0
	for i in range(L):
		for j in range(L):
			en += spins[i,j] * (spins[ (i+1) % L,j] + spins[i, (j + 1) % L])
	return -en

def total_pottsenergy():
	en = 0
	for i in range(L):
		for j in range(L):
			en += pottsenergy( [i,j], spins[i,j] )
	return en/2.0	


def dump(vector, fname):
	fout = file(fname, 'w+')
	for elem in vector:
		fout.write( '%d \n' % elem)	
	fout.close()

def	save_config():
	fname = 'spins/pspins.beta=%2.5f#.L=%d' % (beta, L)
	np.save(fname, spins)

def load_config():
	global spins
	fname = 'spins/pspins.beta=%2.5f#.L=%d' % (beta, L)
	spins = np.load(fname + '.npy')

def dump_spins():
	fname = 'density/pspin_density.beta=%2.5f#.L=%d' % (beta, L)
	fout = file(fname, 'w+')
	for i in range(L):
		for j in range(L):
			fout.write('%d \t %d \t %d \n' % (i,j,spins[i,j]))
	fout.close()


def thermalize(steps = 10**6, _reuse_spins = False):
	if _reuse_spins:
		last_energy = total_pottsenergy()
		last_magnetization = spins.sum()
	else:
		[last_energy, last_magnetization] = initialize_potts()
		dump_spins()
	
	counter = 0

	energies = []
	magnetization  = []
	for x in range(steps):
		[dmagn, denergy] = potts_step()

		if counter % (5 * 10**5) == 0:
			getstatusoutput("kinit -R")			
			print 'aklog=' + getstatusoutput("aklog")[1]  
		counter += 1

		last_energy += denergy
		last_magnetization += dmagn
		energies.append(last_energy)
		magnetization.append( last_magnetization )
	
	fname = '/tmp/Ising2D/pthermalization.q=%d.beta=%2.5f#.L=%d' % (q, beta, L)
	fout = file(fname, 'w+')
	for i in range(steps):
		# step \t E \t M
		if (i % 10**5) or not _reuse_spins:
			fout.write('%d \t %2.2f \t %2.2f \n' % (i, energies[i], magnetization[i]) )
	fout.close()
	save_config()	



def measure(measurement_step = 2 * 10**5, measurements = 15, _continue = False):
	global beta, L, spins

	if not _continue:
		load_config()

	# Initialize temp variables	
	last_energy = total_pottsenergy()
	last_magnetization = spins.sum()
	energies = []
	magnetization  = []
	energies_squared = []
	mag_squared = []
	abs_mag = []

	# Begin measuring on a thermalized state
	for x in range( measurements ):
		temp_energy = 0
		temp_magnetization = 0
		temp_e2 = 0.0
		temp_mag2 = 0.0
		temp_absmag = 0.0
		counter = 0

		print '+',
		for y in range(measurement_step):
			[dmagn, denergy] = potts_step()
			# Update energy and magnetization to current values
			last_energy += denergy
			last_magnetization += dmagn
			# Add the current energy and magnetization (simple and squared) to a running total
			temp_energy += last_energy
			temp_e2 += last_energy ** 2
			temp_magnetization += last_magnetization
			temp_mag2 += last_magnetization ** 2	
			# Absolute value of magnetization
			temp_absmag += abs(last_magnetization)		
		
		# Renew kerberos tickets
		#if counter % (10**4) == 0:
		getstatusoutput("aklog") 
		# counter += 1
		


		# Divide running total by number of steps in one measurement
		energies.append(temp_energy * 1.0  / measurement_step)
		magnetization.append( temp_magnetization * 1.0 / measurement_step )
		energies_squared.append( temp_e2 * 1.0 / measurement_step)
		mag_squared.append( temp_mag2 * 1.0 / measurement_step)
		abs_mag.append( temp_absmag * 1.0 / measurement_step )
	
	# Done measuring
	fname = 'data/pdata.q=%d.beta=%2.5f#.L=%d' % (q, beta, L)
	count = len(energies) * 1.0
	en = sum(energies) / count
	en2 = sum(energies_squared) / count
	mag = sum(magnetization) / count
	mag2 = sum(mag_squared) / count
	absmag = sum(abs_mag) / count

	cv = ( en2 - en**2) / (L**2 ) * beta**2
	chi = (mag2 - mag**2) / (L**2) * beta
	chip = (mag2 - absmag**2) / (L**2) * beta	

	fout = file(fname, 'w+')
	#fout.write('', ) 
	fout.write('%5.4f \t %4.4f \t %4.4f \t %4.4f \t %4.4f \n' % (1/beta, en/ L**2, en2 / L**2, absmag / L**2, mag2 / L**4 ) )
	fout.write('%4.4f \t %8.4f \t %4.4f \t %8.4f \t %4.4f \n' % (en, en2, mag, mag2, absmag ))
	fout.write('# \n')

	for i in range(measurements):
		# step \t E \t E^2 \t M \t M^2 \t |M|
		fout.write('%d \t %4.4f \t %8.4f \t %4.4f \t %8.4f \t %4.4f \n' % (i, energies[i], energies_squared[i], magnetization[i], mag_squared[i], abs_mag[i] ))
	fout.close()
	print '\n'
	
def evolve(betas, _L = None):
	global beta, L

	if _L != None: 
		L = _L
	print 'Working with L = %d ' % L
	
	getstatusoutput("aklog") 

	for val in betas:
		beta = val
		print 'Current beta = %2.6f' % beta
		reuse = (beta != betas[0])  # If it's not the first beta, then we should reuse the lattice
		thermalize(thermo_steps, _reuse_spins = reuse )
		getstatusoutput("aklog") 
		measure(measurements = 20, _continue = reuse )
		dump_spins()

#betas = [1 / 0.35, 1/0.7, 1/0.8]
betas = [ 0.0005, 0.001, 0.01, 0.1, 0.25, 0.3, 0.4, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.5, 1.0, 1.15, 1.25, 1.42, 1.666, 1.8,  10.0]


thermo_steps = 10 * 10**5

if __name__ == "__main__":
	global L, q, betas
	print getstatusoutput("mkdir -p /tmp/Ising2D")[1]
	if len(sys.argv) > 1:
		print sys.argv
		q = int(sys.argv[1])
		L = int(sys.argv[2])
	
	if len(sys.argv) > 2:
		print 'Working with q=%d \t L=%d' % (q, L)
		
	if q == 3:
		betas = [ 1 / 0.95, 1.0, 1 / 1/1.1, 1/ 0.9, 1/ 1.2, 1/1.3]
	else:
		betas = [0.5, 1.0, 1.388, 1.398, 1.408, 1.4184, 1.4285, 1.4388, 1.449]

	evolve(betas)




