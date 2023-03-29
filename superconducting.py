import numpy as np
import cellconstructor as CC
import cellconstructor.ForceTensor
from cellconstructor.Units import *

RY_EV = 27.21139613179/2.0
KB_EV = 8.617333262145e-5

def interpolate_dyn(dense, coarse, dfpt_coarse, dfpt_dense, nqirr_coarse, nqirr_dense, support):
	sscha_coarse_dyn = CC.Phonons.Phonons(coarse, nqirr_coarse)
	if(support):
		dfpt_coarse_dyn = CC.Phonons.Phonons(dfpt_coarse, nqirr_coarse)
		dfpt_dense_dyn = CC.Phonons.Phonons(dfpt_dense, nqirr_dense)
		dyn = sscha_coarse_dyn.Interpolate(coarse_grid = sscha_coarse_dyn.GetSupercell(), fine_grid = dfpt_dense_dyn.GetSupercell(), support_dyn_coarse = dfpt_coarse_dyn, support_dyn_fine = dfpt_dense_dyn, symmetrize = True)
		dyn.save_qe(filename=dense)
	else:
		dyn = sscha_coarse_dyn.Interpolate(coarse_grid = sscha_coarse_dyn.GetSupercell(), fine_grid = dfpt_dense_dyn.GetSupercell(), symmetrize = True)
		dyn.save_qe(filename=dense)
	return dyn	

def check_if_opposite_true(D, Z, W, T, lnm, lnm_mu, thr):
	D_old = D.copy()
	v1 = np.zeros_like(W)
	v2 = np.zeros_like(W)
	for i in range(len(W)):
		v1[i] = W[i]/np.sqrt(W[i]**2 + D[i]**2)
		v2[i] = D[i]/np.sqrt(W[i]**2 + D[i]**2)
	for i in range(len(W)):
		suma = 0.0
		for j in range(len(W)):
			suma += lnm[i,j]*v1[j]
		Z[i] = 1.0 + np.pi*T*suma/W[i]
		suma = 0.0
		for j in range(len(W)):
			suma += lnm_mu[i,j]*v2[j]
		D[i] = suma*np.pi*T/Z[i] 
	diff = np.linalg.norm(D - D_old)/np.linalg.norm(D_old)
	if(diff <= thr):
		return True
	else:
		return False
	

def solve_isotropic(a2f, omega, mu, wcut, T, max_iter, delta0, thr, mixing):
	diff = 1.0
	w = []
	n = 0
	while(float(2*n + 1)*np.pi*T < wcut):
		if(n == 0):
			w.append(float(2*n + 1)*np.pi*T)
		else:
			w.append(float(2*n + 1)*np.pi*T)
			w.append(float(-2*n + 1)*np.pi*T)
		n = n + 1
	print('Number of matsubara frequencies at temperature ' + format(T*RY_TO_EV/KB_EV, '.2f') + ' is ' + str(len(w)) + '.')
	w.sort()
	w = np.array(w)
	index = int(len(w)/2) + 1
	for i in range(1, len(w)):
		if(w[i] > 0.0 and w[i] < 0.0):
			index = i
			break
	delta1 = np.zeros(len(w))
	delta1[index] = delta0
	z1 = np.zeros(len(w))
	iteration = 1
	lnm = np.zeros((len(w), len(w)))
	for j in range(len(w)):
		for k in range(1, len(omega)):
			lnm[0,j] += 2.0*omega[k]*a2f[k]/((w[0] - w[j])**2 + omega[k]**2)*(omega[k] - omega[k-1])
		lnm[j,0] = lnm[0,j]
	for i in range(1, len(w)):
		for j in range(i, len(w)):
			if(i == j):
				lnm[i,i] = lnm[0,0]
			else:
				difference = j - i
				lnm[i,j] = lnm[0, difference]
				lnm[j,i] = lnm[i,j]
	lnm_mu = lnm - mu#*np.diag(np.ones(np.shape(lnm)))
	while(diff > thr and iteration < max_iter):
		delta2 = delta1.copy()
		v1 = w/np.sqrt(w**2 + delta1**2)
		v2 = delta1/np.sqrt(w**2 + delta1**2)
		z1 = 1.0 + np.dot(lnm, v1)*np.pi*T/w
		delta1 = np.dot(lnm_mu, v2)*np.pi*T/z1
		delta1 = (1.0 - mixing)*delta1 + mixing*delta2
		diff = np.linalg.norm(delta1 - delta2)/np.linalg.norm(delta2)
		iteration += 1
	if(iteration >= max_iter):
		print('Calculation did not converge!')
	else:
		print('Converged in ' + str(iteration) + ' iterations. The difference was: ' + format(diff, '.3e'))
		if(delta1[index] < 0.0):
			print('Negative gap, hmmm. Checking if the opposite solution satisfies self-consistency ...')
			same = check_if_opposite_true(-1.0*delta1, z1, w, T, lnm, lnm_mu, thr)
			if(same):
				print('Opposite solution satisfies self-consistency! Using opposite solution.')
				delta1 = -1.0*delta1
	return w, delta1, z1, index

def get_qpoint_ids(dyn, qpts):

	ids = np.zeros(len(qpts), dtype=int)
	for iqpt in range(len(qpts)):
		found = False
		for jqpt in range(len(dyn.q_tot)):
			qpt = np.dot(dyn.q_tot[jqpt], dyn.structure.unit_cell.T)
			diff = qpt - qpts[iqpt]
			if(np.linalg.norm(diff - np.rint(diff)) < 5.0e-5):
				ids[iqpt] = jqpt
				found = True
				break
		if(not found):
			print('Could not find q point: ')
			print(qpts[iqpt])
			print('It probably is not in the commensurate grid of DYN supercell.')
	return ids

def gaussian(x, x0, smearing):
	return np.exp(-0.5*(x-x0)**2/smearing**2)/np.sqrt(2.0*np.pi)/smearing
		
def calculate_a2f(dyn, qpts, weights, dos, elph, smearing, nom):

	freqs = []
	eigs = []
	
	qpts_ids = get_qpoint_ids(dyn, qpts)
	for iq in range(len(qpts_ids)):
		curr_freqs, curr_pols = dyn.DyagDinQ(qpts_ids[iq])
		freqs.append(curr_freqs)
		eigs.append(curr_pols.T)
	eigs = np.array(eigs)
	for iq in range(len(qpts_ids)):
		for iat in range(dyn.structure.N_atoms):
			eigs[iq][:,3*iat:3*(iat + 1)] = eigs[iq][:,3*iat:3*(iat + 1)]/np.sqrt(dyn.structure.masses[dyn.structure.atoms[iat]])
	freqs = np.array(freqs)
	lambdas = np.zeros_like(freqs)
	freq_max = np.amax(freqs)*1.1
	omega = (np.arange(nom, dtype = float) + 1.0)/float(nom)*freq_max
	aq = np.zeros((len(qpts_ids), len(freqs[0]), len(omega)))
	pq_dos = np.zeros_like(aq)
	aF = np.zeros(len(omega))
	pdos = np.zeros_like(aF)
	mat_prod = np.zeros((len(qpts_ids), len(freqs[0])))
	for iq in range(1, len(qpts_ids)):
		print(str(iq + 1) + ' Q point: ' + format(qpts[iq][0], '.8f') + 3*' ' + format(qpts[iq][1], '.8f') + 3*' ' + format(qpts[iq][2], '.8f'))
		for iband in range(len(freqs[iq])):
			ph_elem = np.dot(eigs[iq, iband], np.dot(elph[iq], eigs[iq, iband].conj()))
			if(ph_elem.imag != 0.0 and np.abs(ph_elem.real/ph_elem.imag) < 1.0e6):
				print('Imaginary part of the ph_elem is too large! ' + format(ph_elem.real/ph_elem.imag, '.2e'))
			mat_prod[iq, iband] = np.dot(eigs[iq, iband].conj(), np.dot(elph[iq], eigs[iq, iband])).real
			lambdas[iq, iband] = mat_prod[iq, iband]/2.0/freqs[iq, iband]**2/dos
			pq_dos[iq, iband] = float(weights[iq])*np.array([gaussian(freqs[iq, iband], w, smearing) for w in omega])
			aq[iq, iband] = mat_prod[iq, iband]/4.0/freqs[iq, iband]/dos*pq_dos[iq, iband]
			aF += aq[iq, iband]
			pdos += pq_dos[iq, iband]
			print('Frequency (1/cm): ' + format(freqs[iq, iband]*RY_TO_CM, '.8f') + 3*' ' + 'Lambda: ' + format(lambdas[iq, iband], '.8f') + ' (' + format(2.0*np.sum(aq[iq,iband]/omega)*omega[0]/float(weights[iq]), '.8f')+ ')')
		print(' ')
			
	aF = aF/float(np.sum(weights) - 1)
	pdos = pdos/float(np.sum(weights) - 1)
	return aF, pdos, aq, pq_dos, lambdas, omega

