import numpy as np
import os

def which_q(filename):
	if(os.path.exists(filename)):
		numq = 0
		qpts = []
		rprim = np.zeros((3,3))
		with open(filename, 'r') as infile:
			content = infile.readlines()
			for i in range(len(content)):
				if('Basis vectors' in content[i]):
					for j in range(3):
						line = content[i + j + 1].strip().split()
						for k in range(3):
							rprim[j,k] = float(line[k])
				elif('q = (' in content[i]):
					line = content[i].strip().split()
					qpt = np.zeros(3)
					for j in range(3):
						qpt[j] = float(line[j + 3])
					if(np.any(rprim != 0.0)):
						qpt = np.dot(qpt, rprim.T)
					qpts.append(qpt)
					numq += 1
		return qpt, numq - 1, qpts[:-1]
	else:
		raise RuntimeError('File ' + filename + 'does not exist!')
	

def get_data_from_elph_file(filename, natom, skip):
	curr_smear = []
	curr_elph = []
	curr_dos = []
	with open(filename, 'r') as infile:
		content = infile.readlines()
		if(len(content)%skip != 0):
			raise RuntimeError('Unexpected number of lines in ' + filename + ' file.') 
		else:
			for i in range(int(np.rint(float(len(content)/float(skip))))):
				line = content[i*skip].strip().split()
				curr_smear.append(float(line[0]))
				line = content[i*skip + 1].strip().split()
				curr_dos.append(float(line[0]))
				smear_elph = np.zeros((3*natom, 3*natom), dtype = complex)
				iline = 0
				for iband in range(3*natom):
					for jband in range(3*natom):
						line = content[i*skip + 2 + iline].strip().split(',')
						if(len(line) == 2):
							smear_elph[iband, jband] = complex(float(line[0].replace('(', '')), float(line[1].replace(')', '')))
						else:
							print(line)
							raise RuntimeError('Could not read in electron-phonon coupling! ')
						iline += 1
				curr_elph.append(smear_elph)
	return curr_smear, np.array(curr_dos), curr_elph

def read_elph(prefix, nqirr, natom):
	qpts = np.zeros((nqirr, 3))
	qstar = []
	weights = np.zeros(nqirr)
	elph = []
	skip = 2 + (3*natom)**2
	for iqpt in range(nqirr):
		qpts[iqpt], weights[iqpt], qs = which_q(prefix + str(iqpt + 1))
		qstar.append(qs)
		filename = prefix + str(iqpt + 1) + '.elph.d.mat.' + str(iqpt + 1)
		curr_smear, curr_dos, curr_elph = get_data_from_elph_file(filename, natom, skip)
		if(iqpt == 0):
			smearings = curr_smear.copy()
			dos = curr_dos.copy()
		else:
			if(smearings != curr_smear):
				print('Smearing at ' + str(iqpt + 1) + ' q point is not equal to smearing at Gamma!')
				print('Gamma: ')
				print(smearings)
				print(str(iqpt + 1) + ': ')
				print(curr_smear)
			elif(np.linalg.norm(curr_dos - dos) > 1.0e-6):
				print('DOS at ' + str(iqpt + 1) + ' q point is not equal to DOS at Gamma!')
				print('Gamma: ')
				print(dos)
				print(str(iqpt + 1) + ': ')
				print(curr_dos)
		elph.append(curr_elph)
	return qpts, smearings, dos, np.array(elph), weights, qstar
				
