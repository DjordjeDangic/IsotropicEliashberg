from __future__ import print_function
from __future__ import division

# Import the modules to read the dynamical matrix
import numpy as np
import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.ForceTensor
import parse_elph
import superconducting
from cellconstructor.Units import *
import time

start_time = time.time()
KB_EV = 8.617333262145e-5
dfpt_dense_dyn_prefix = './H.dyn'         # prefix for DFPT dynamical matrices and deformation potential files on dense grid
dfpt_coarse_dyn_prefix = './H_coarse.dyn' # prefix for DFPT dynamical matrices and deformation potential files on coarse grid (the same grid as SSCHA calculation)
coarse_dyn_prefix = './sscha_coarse.dyn'  # prefix for SSCHA dynamical matrices
dense_dyn_prefix = './sscha_dense.dyn'    # prefix for interpolated SSCHA dynamical matrices
interpolate_w_support = True              # If True will interpolate only anharmonic part of dynamical matrices
nqirr = 30                                # Number of irreducible q points in dense grid
nqirr_coarse = 18                         # Number of irreducible q points in coarse grid

# Interpolate SSCHA dynamical matrices
dyn = superconducting.interpolate_dyn(dense_dyn_prefix, coarse_dyn_prefix, dfpt_coarse_dyn_prefix, dfpt_dense_dyn_prefix, nqirr_coarse, nqirr, interpolate_w_support)
freqs, pols = dyn.DiagonalizeSupercell()
natom = dyn.structure.N_atoms

supercell_matrix = dyn.GetSupercell()
# Read in electron-phonon coupling 
qpts, smearings, dos, elph, weights, qstar = parse_elph.read_elph(dfpt_dense_dyn_prefix, nqirr, natom)

nom = 2000                                # Number of frequency points in \alpha^2F
start_temp = 0.0                          # Starting temperature for solving ME equations
sigma = 0.008                             # Double delta smearing used in electron-phonon calculation
smearing = 10.00/RY_TO_CM                 # Smearing for Gaussian that approximate phonon spectral functions
max_temp = 500.0                          # Maximum temperature for which to calculate ME equations
ntemp = 100                               # Number of temperature steps to take in range (start_temp, max_temp)
w_cut = max(freqs)*10.0                   # Cutoff for Matsubara frequencies
print('Cutoff for Matsubara frequencies is: ' + format(w_cut, '.5f') + ' Ry.')
mu = 0.16                                 # Reduced Coulomb potential
thr = 1.0e-4                              # Self-consistency threashold for superconducting gap
mix = 0.2                                 # Mixing parameter for self-consistency
max_iter = 10000                          # Maximum number of iterations for the solution of ME equations


# Check if requested double delta smearing is in the electron-phonon files
found_smearing = False
for i in range(len(smearings)):
	if(np.abs(smearings[i] - sigma) < 1.0e-8):
		ism = i
		found_smearing = True
		break

if(found_smearing):
	print('Calculating a2F...')
	aF, pdos, aq, pq_dos, lambdas, omega = superconducting.calculate_a2f(dyn, qpts, weights, dos[ism], elph[:,ism,:,:], smearing, nom)
        # Writing \alpha^2F
	with open('a2f', 'w+') as outfile:
		for i in range(nom):
			outfile.write(3*' ' + format(omega[i], '.12e'))
			outfile.write(3*' ' + format(aF[i], '.12e'))
			outfile.write(3*' ' + format(pdos[i], '.12e') + '\n')
	lambda_tot = 2.0*np.sum(aF/omega)*omega[0]
	lambda_tot1 = np.average(np.sum(lambdas, axis = 1)[1:], weights = weights[1:])

        # These two numbers should be really close. Increase nom if they are not!
	print('Lambda final from a2F: ' + format(lambda_tot, '8f'))
	print('Lambda average over q vectors: ' + format(lambda_tot1, '8f'))

	T_mcm = omega[-1]*RY_TO_EV/KB_EV/1.45*np.exp(-1.04*(1.0 + lambda_tot)/(lambda_tot*(1.0-0.62*mu) -mu))/1.1
	print('McMillan\'s estimate of transition temperature: ', T_mcm)

	temps = (np.arange(ntemp, dtype = float) + 1)*(max_temp - start_temp)/float(ntemp) + start_temp
	w_cut = w_cut

	delta0 = T_mcm*3.52/2.0*KB_EV/RY_TO_EV
	print('Initial guess for the gap (Ry): ', format(delta0, '.2e'))
	delta = []
	Z = []
	indices = []
	for i in range(ntemp):
		print('Calculating for temperature: ' +  format(temps[i], '.2f') + ' K!')
		if(i == 0):
			w_i, delta_i, z_i, index = superconducting.solve_isotropic(aF, omega, mu, w_cut, temps[i]*KB_EV/RY_TO_EV, max_iter, delta0, thr, mix)
		else:
			w_i, delta_i, z_i, index = superconducting.solve_isotropic(aF, omega, mu, w_cut, temps[i]*KB_EV/RY_TO_EV, max_iter, delta[-1][indices[-1]], thr, mix)

		print('Solved isotropic equations!')
		delta.append(delta_i)
		Z.append(z_i)
		indices.append(index)
		with open('gap_at_' + format(temps[i], '.2f'), 'w+') as outfile:
			for j in range(len(w_i)):
				outfile.write(3*' ' + format(w_i[j], '.8e'))
				outfile.write(3*' ' + format(delta_i[j], '.8e'))
				outfile.write(3*' ' + format(z_i[j], '.8e') + '\n')
		if(np.abs(delta_i[index]) < delta0/1000.0):
			print('The value of superconducting gap is one thousand times smaller than predicted in Allen-Dynes. Stopping calculation ...')
			break
			

	with open('GAP', 'w+') as outfile:
		outfile.write('#    T(K)')
		outfile.write('      GAP (Ry)  ')
		outfile.write('        Z   \n')
		for i in range(len(delta)):	
			outfile.write(3*' ' + format(temps[i], '.2f'))
			outfile.write(3*' ' + format(delta[i][indices[i]], '.8e'))
			outfile.write(3*' ' + format(Z[i][indices[i]], '.8e') + '\n')
		for j in range(i+1, ntemp):
			outfile.write(3*' ' + format(temps[j], '.2f'))
			outfile.write(3*' ' + format(0.0, '.8e'))
			outfile.write(3*' ' + format(Z[-1][indices[-1]], '.8e') + '\n')

	print('Calculation finished in: ' + format(time.time() - start_time, '.1f') + ' seconds!')
			
else:
	print('Could not find requested smearing!') 
