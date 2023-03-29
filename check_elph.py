from __future__ import print_function
from __future__ import division

# Import the modules to read the dynamical matrix
import numpy as np
import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.ForceTensor
import cellconstructor.ThermalConductivity
import cellconstructor.SSCHA_phonopy_parser
import parse_elph
import superconducting
from cellconstructor.Units import *
import time

start_time = time.time()
KB_EV = 8.617333262145e-5
dyn_prefix = 'H.dyn'
nqirr = 163
comm_nqirr = 18

dyn = CC.Phonons.Phonons('sscha_dense.dyn', nqirr)
comm_dyn = CC.Phonons.Phonons('sscha_coarse.dyn', comm_nqirr)
natom = dyn.structure.N_atoms

supercell_matrix = dyn.GetSupercell()
qpts, smearings, dos, elph, weights, qstar = parse_elph.read_elph(dyn_prefix, nqirr, natom)
sigma = 0.012
smearing = 10.00/RY_TO_CM
nom = 1000
max_temp = 500.0
ntemp = 500
w_cut = 0.15 # Ry
mu = 0.16 
thr = 1.0e-4
mix = 0.2
max_iter = 10000
third_order = 'd3_realspace_sym.npy'
scale = [24, 24, 24]
no_mode_mixing = True


found_smearing = False
for i in range(len(smearings)):
	if(np.abs(smearings[i] - sigma) < 1.0e-8):
		ism = i
		found_smearing = True
		break
if(found_smearing):
	freqs = []
	eigs = []
	
	qpts_ids = superconducting.get_qpoint_ids(dyn, qpts)
	for iq in range(len(qpts_ids)):
		curr_freqs, curr_pols = dyn.DyagDinQ(qpts_ids[iq])
		freqs.append(curr_freqs)
		eigs.append(curr_pols.T)
	eigs = np.array(eigs)
	for iq in range(len(qpts_ids)):
		print('Q point: ' + str(iq + 1))
		for iat in range(dyn.structure.N_atoms):
			eigs[iq][3*iat:3*(iat + 1)] = eigs[iq][3*iat:3*(iat + 1)]#/np.sqrt(dyn.structure.masses[dyn.structure.atoms[iat]])
		mat_prod = [np.dot(eigs[iq, iband].conj(), np.dot(elph[iq, ism], eigs[iq, iband])).real for iband in range(3*dyn.structure.N_atoms)]
		print('Projected: ')
		print(mat_prod)
		print('Diagonal: ')
		print(np.diag(elph[iq,ism]))
