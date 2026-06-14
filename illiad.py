"""
Copyright (c) 2026 Danila Chernyavskiy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union
from tabulate import tabulate
from shelxfile import Shelxfile
import os
import sys
import argparse

Header = r"""
 ┌─────────────────────────────────────────────────────────────────┐ 
 │  ILLIAD - Precise Layered Molecular Packing Metrics Evaluation  │  
 │  (C) 2026 D.R. Chernyavskiy                           v2026/a1  │ 
 └─────────────────────────────────────────────────────────────────┘ 
 """

Footer = r"""

 Please include the required citations if this program yielded useful results:
 [1]	<LINK FOR ILLIAD>
 [2]	Cryst. Growth. Des. 2024, 24, 9849-9856. doi: 10.1021/acs.cgd.4c01174
 [3]	J. Phys. Chem. A 1964, 68, 441-451. doi: 10.1021/j100785a001
 [4]	J. Phys. A: Math. Gen. 2004, 37, 11591-11601. doi: 10.1088/0305-4470/37/48/005

"""

# Set up the parsing environment
parser = argparse.ArgumentParser(description = 'ILLIAD - Precise Layered Molecular Packing Metrics Evaluation, (c) 2026, D.R. Chernyavskiy')

parser.add_argument('-hkl', '--hkl', nargs = 3, type = int, help = 'Miller indices of the layer plane')
parser.add_argument('-m', '--mol', type = int, help = 'number of molecules on a layer per unit cell')
parser.add_argument('-r', '--resolution', type = int, default = 300, help = 'vdW surface cloud resolution')
parser.add_argument('-q', '--quiet', default = False, action="store_true", help = 'quiet mode without any verbose')
parser.add_argument('--sect', type = bool, default = False, help = 'plot the layer section of the molecule')
parser.add_argument('--surf', type = bool, default = False, help = 'plot the vdW surface')
parser.add_argument('--scld', type = bool, default = False, help = 'plot the vdW surface cloud points')
parser.add_argument('-p', '--project', type = str, help = 'SHELX RES file project name')
args = parser.parse_args()
quiet = args.quiet

if args.resolution == '' or args.resolution == None:
	args.resolution = 300

print(Header)

atomColors = {
	'H': 'gray',
	'C': 'black',
	'N': 'blue',
	'O': 'red',
	'F': 'green',
	'P': 'orange',
	'S': 'yellow',
	'Cl': 'green'
}

vdwBondi = {
	'H': 1.20,
	'Li': 1.81,
	'C': 1.70,
	'N': 1.55,
	'O': 1.52,
	'F': 1.47,
	'Na': 2.27,
	'Mg': 1.73,
	'Si': 2.10,
	'P': 1.80,
	'S': 1.80,
	'Cl': 1.75,
	'K': 2.75,
	'Ga': 1.87,
	'As': 1.85,
	'Se': 1.90,
	'Br': 1.83,
	'In': 1.93,
	'Sn': 2.17,
	'Te': 2.06,
	'I': 1.98,
	'Tl': 1.96,
	'Pb': 2.02
}


def recombine_points_for_scatter(points):
	xs = [point[0] for point in points]
	ys = [point[1] for point in points]
	zs = [point[2] for point in points]
	return [xs,ys,zs]

def recombine_points_for_scatter2d(points):
	xs = [point[0] for point in points]
	ys = [point[1] for point in points]
	return xs,ys

def get_A_matrix(cell):
	"""
	See frac2cart comment about A matrix. 
	"""
	s1, s2, s3 = np.sin(cell[3]), np.sin(cell[4]), np.sin(cell[5])
	c1, c2, c3 = np.cos(cell[3]), np.cos(cell[4]), np.cos(cell[5])
	a, b, c, V = cell[0], cell[1], cell[2], get_tensor_V(cell)

	A = np.array([
		[a, b*c3, c*c2],
		[0, b*s3, c*(c1-c2*c3)/s3],
		[0, 0, V/(a*b*s3)]
		])
	return A

def get_G_matrix(cell):
	"""
	Calculate the covariant metric tensor from unit cell, G 
	"""
	[a, b, c, alpha, beta, gamma] = cell

	# calculate the covariant metric tensor G and its inverse, Ginv
	G = np.array([
	[a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
	[a*b*np.cos(gamma), b**2, b*c*np.cos(alpha)],
	[a*c*np.cos(beta), b*c*np.cos(alpha), c**2]
	])
	return G

def get_tensor_V(cell):
	return np.sqrt(np.linalg.det(get_G_matrix(cell)))

def frac2cart(fpoint, cell):
	"""
	Converting the fractional coordinates point to a cartesian one.
	The linear form of transformation is r = A*rho,
	where r is the cartesian vector, rho is the fractional one.
		a 	b cos(gamma)	c cos(beta)
	A = 0 	b sin(gamma) 	c*(cos(alpha)-cos(beta)cos(gamma))/sin(gamma)
		0 	0 				V/(ab sin(gamma))

	V is the volume of the unit cell, expressed as
	V = abc*sqrt[1-cos^2(alpha)-cos^2(beta)-cos^2(gamma)+2cos(alpha)cos(beta)cos(gamma)]
	We will use the CIF calculated volume for the ease of calculations.

	cell		[a, b, c, alpha, beta, gamma, V]
	"""
	A = get_A_matrix(cell)
	return np.dot(A, fpoint)

def hklpln(hkl, cell): 
	global quiet

	"""
	Calculate the normal vector, generated by plane from Miller indeces
	"""
	A = get_A_matrix(cell)
	Ainv = np.linalg.inv(A)
	normal_vector = Ainv.T @ np.array(hkl)
	normal_vector /= np.linalg.norm(normal_vector)
	normal_toprint = [round(float(orth), 4) for orth in normal_vector]
	if not quiet:
		print(f'[HKLV] Normal vector to {hkl} is {normal_toprint}')
	return normal_vector

def get_centroid(coords_cart):
	return np.mean(coords_cart, axis = 0)

def mpln(coords):
	global quiet
	"""
	Calculation of an LS-plane using singular value decomposition (SVD),
	i.e. making the somewhat exact analogue to MPLN in ShelXP

	coord 		atomic coordinates array (in Ang)

	output		a, b, c 	normal vector projections
				cent 		centroid point
	"""
	centroid = get_centroid(coords)
	ccoords = coords - centroid
	_, _, vh = np.linalg.svd(ccoords)
	normal_vector = vh[-1, :]

	if not quiet:
		print('[MPLN] Centroid: \t\t\t\t', centroid)
		print('[MPLN] Plane normal vector: \t', normal_vector)

	return normal_vector

def construct_fibonacci_sphere(radius, resolution):
	"""
	Fibonacci sphere implementation

	resolution		surface density of points on a sphere,
					measured by Ang^(-2)
	radius			sphere radius

	The total number of points is resoluion * sphere surface area,
	rounded to the nearest integer (upper limit - ceil) 
	Output - np.ndarray of points on a sphere

	For 8192 triangles we would need 4167 points on a sphere,
	hense the resolution for a unit sphere should be 
	4167 / 4*pi = approx 332 Ang^(-2).
	We will round it to approx. 300.
	"""
	n = np.ceil(4 * np.pi * radius**2 * resolution)
	indices = np.arange(0, n, dtype = float) + 0.5
	phi = np.arccos(1 - 2*indices/n)
	theta = np.pi * (1+5**0.5) * indices

	x,y,z = radius*np.sin(phi)*np.cos(theta), radius*np.sin(phi)*np.sin(theta), radius*np.cos(phi)
	return np.stack([x,y,z], axis = 1)

def generate_vdw_points(coords, types, resolution, custom_radii = False):
	global quiet
	if not quiet:
		print(f'[GENP] Using sphere point resolution of {resolution} Ang^(-2)...')

	"""
	Generate van der Waals surface points using tabular (Bondi)
	or custom (effective) radii.

	coords			n by 3 array of atom coordinates in Ang
	types 			n by 1 array of atom types (as symbols)
	resolution		point density, used in construct_fibonacci_sphere
	custom_radii	boolean for custom radii usage (False by default)
	"""
	coords = np.array(coords)
	natoms = len(coords)
	points = []

	for i in range(natoms):
		atompts = construct_fibonacci_sphere(vdwBondi[types[i]], resolution) + coords[i]

		# checking for neighboring points,
		# as spheres themselves intersect
		neighbors = []
		for j in range(natoms):
			if i == j:
				continue
			d2 = np.sum((coords[i] - coords[j])**2)
			if d2 < (vdwBondi[types[i]] + vdwBondi[types[j]])**2:
				neighbors.append(j)

		# checking for each point whether it acccidentally
		# lies inside another atom
		for point in atompts:
			is_point_accessable = True
			for j in neighbors:
				if np.sum((point - coords[j])**2) < vdwBondi[types[j]]**2:
					is_point_accessable = False
					break
			if is_point_accessable:
				points.append(point)
	if not quiet:
		print('[FIBO] Constructed points for each atom by Fibonacci method')
		print('[GENP] Omitted the internal points')
		print(f'[GENP] The vdW surface (Bondi radii) consists of {len(points)} points')
	return np.array(points)

def ExtractCoordsFromRESFile(ProjectName):
	global quiet
	# As in CIF file, we have to return the cell angles in RADIANS.
	res = ProjectName + '.res'
	shx = Shelxfile(verbose = True)
	if os.path.isfile(res):
		shx.read_file(res)
	else:
		print(f'[REAP] ERROR: could not read {res}, exiting...')
		sys.exit(1)
	cell = list(shx.cell)
	cell[3], cell[4], cell[5] = cell[3]/360*2*np.pi, cell[4]/360*2*np.pi, cell[5]/360*2*np.pi,
	Z = int(list(shx.zerr)[1])
	atom_types = []
	coords_cart = []
	for atom in shx.atoms.nameslist:
		a = shx.atoms.get_atom_by_name(atom)
		atom_types.append(a.element)
		coords_cart.append(a.cart_coords)
	if not quiet:
		print(f'[REAP] Read {res} successfully ({round(os.path.getsize(res)/1024)} kB)')
	return cell, coords_cart, atom_types, Z

def get_orthonormal_basis(norm):
	if abs(norm[0]) < 0.9:
		u = np.cross(norm, [1,0,0])
	else:
		u = np.cross(norm, [0,1,0])
	u = u/np.linalg.norm(u)
	v = np.cross(norm, u)
	v = v/np.linalg.norm(v)

	return u,v


def plot_plane(ax, norm, cent, size = 2, resolution = 2):
	""" 
	Plotting the 3d-plane from the normal vector, passing through the centroid
	norm 		normal vector
	cent 		centroid
	size 		plane linear dimension in Ang
	resolution	plane resolution in Ang-1
	"""

	u,v = get_orthonormal_basis(norm)

	t = np.linspace(-size, size, resolution)
	tt, ss = np.meshgrid(t, t)

	X = cent[0] + u[0]*tt + v[0]*ss 
	Y = cent[1] + u[1]*tt + v[1]*ss 
	Z = cent[2] + u[2]*tt + v[2]*ss 

	ax.plot_surface(X, Y, Z, alpha = 0.5, color = 'red')


def get_hkl_section_surface_area(hkl, cell):
	global quiet
	"""
	Calculate the area of the section generated by (h k l) Miller indeces through a unit cell.
	hkl 		3x1 array of Miller indeces
	cell 		= [a, b, c, alpha, beta, gamma]
	THE ANGLES ARE ALREADY IN RADIANS, NO NEED TO CONVERT THEM
	"""
	G = get_G_matrix(cell)
	Ginv = np.linalg.inv(G)
	g11, g22, g33 = Ginv[0][0], Ginv[1][1], Ginv[2][2]
	g12 = Ginv[0][1]
	g23 = Ginv[1][2]
	g13 = Ginv[0][2]

	# the cross-sectional area S(hkl) = sqrt{|G|*|H|^2}, where
	# |H|^2 = (hkl)_row Ginv (hkl)_column - a regular quadratic form
	# numpy is smart enough to convert hkl to hkl.T on its own
	H2 = hkl @ Ginv @ hkl
	S = np.sqrt(np.linalg.det(G)*H2)
	if not quiet:
		print(f'[HKLS] Sectional area of unit cell by {hkl} plane is {S:.3f} Ang^2')
	return S


def vdw2plane(coords, types, norm, cent):
	global quiet
	"""
	Calculation of points on LS-plane as cross-section points of vdW surface.
	
	Let the distance between each atom and the plane be equal to z, then
	the "effective" vdW radius would be r_eff = sqrt{r_vdW^2 - z^2}.
	For each atom, we will construct a circle with the corresponding radius.
	We then will go through combining the resulting surface in order to calculate
	the needed values for ILMPC

	coords 		atomic coordinates array
	types 		atomic type array 
	norm 		normal vector (via MPLN or HKL)
	cent 		MPLN centroid
	"""

	u,v = get_orthonormal_basis(norm)
	radii = [vdwBondi[type] for type in types]
	natoms = len(coords)
	circles = []
	atoms_proj = []
	
	for atom, r in zip(coords, radii):
		z = abs(np.dot(atom - cent, norm))
		if z >= r:
			reff = 0
		else:
			reff = np.sqrt(r**2 - z**2)
		proj_center = atom - np.dot(atom-cent, norm) * norm
		cu = np.dot(proj_center - cent, u)
		cv = np.dot(proj_center - cent, v)
		atoms_proj.append([cu, cv])
		circles.append((cu, cv, reff))

	if not circles:
		print('!!! The atomic projections onto MPLN resulted in an empty array. !!!')
		print('!!! Returning Null.                                              !!!')
		return 0,0
	polygons = [Point(cu, cv).buffer(reff) for (cu, cv, reff) in circles]
	union = unary_union(polygons)
	if union.is_empty:
		print('!!! The union of atomic vdW sphere projections resulted in an empty array. !!!')
		print('!!! Returning Null.                                                        !!!')
		return 0,0
	area = union.area
	perimeter = union.length
	if not quiet:
		print(f'[VDWP] Projection total area: {area:.3f} Ang^2')
		print(f'[VDWP] Projection total perimeter: {perimeter:.3f} Ang')

	return atoms_proj, union

# -------------------------------------------------------------------------------------------------------------
ProjectName = args.project
if ProjectName == '' or ProjectName == None:
	ProjectName = input('SHELX project name [.res]: ')

cell, coords_cart, atom_types, Z = ExtractCoordsFromRESFile(ProjectName)

hkl = args.hkl
if hkl == '' or hkl == None:
	hkl = [int(i) for i in input('Layer Miller indeces (hkl): ').split()]

mol_num_per_layer = args.mol
if mol_num_per_layer == '' or mol_num_per_layer == None:
	mol_num_per_layer = int(input('Number of molecules per layer in a unit cell: '))

Sl = get_hkl_section_surface_area(hkl, cell)
norm = hklpln(hkl, cell)
cent = get_centroid(coords_cart)

atoms_proj, U = vdw2plane(coords_cart, atom_types, norm, cent)
S = U.area
P = U.length
x,y = U.exterior.xy

if args.sect:
	xa, ya = recombine_points_for_scatter2d(atoms_proj)
	plt.fill(x,y, alpha = 0.5, edgecolor = 'black')
	for i, type in enumerate(atom_types):
		plt.scatter(xa[i], ya[i], s = 200, color = atomColors[type])
	plt.axis('equal')
	plt.show()

vdwpoints = generate_vdw_points(coords_cart, atom_types, resolution = args.resolution)
if not quiet:
	print('[VDWC] Reconstructing surface from vdW point cloud... ', end='')
cloud = pv.PolyData(vdwpoints)
surf = cloud.reconstruct_surface(sample_spacing = 0.1)
if not quiet:
	print('done')

print('\n---------- FINAL RESULTS ----------')
V = get_tensor_V(cell)
sms_cylinder = surf.volume/(2*vdwBondi['C'])
das_cylinder = len(atom_types)/sms_cylinder
das_exact = len(atom_types)/S
pack_coeff = Z*surf.volume/V
d_interlayer = V/Sl
# C(m,s) = 2*sqrt{pi*S(m,s)} for cylindrical approximation
cms = 2*np.sqrt(np.pi*sms_cylinder)
ilmpc = mol_num_per_layer*sms_cylinder / Sl
ilmpc2 = mol_num_per_layer*S / Sl


table = [
['V(vdW), Ang^3', round(surf.volume, 3), '---'],
['S(m,s), Ang^2', round(sms_cylinder, 3), round(S, 3)],
['C(m,s), Ang', round(cms, 3), round(P, 3)],
['d(a,s), Ang^(-2)', round(das_cylinder, 3), round(das_exact, 3)],
['ILMPC', round(ilmpc, 3), round(ilmpc2, 3)],
['PC', round(pack_coeff, 3), '---'],
['d(IL), Ang', round(d_interlayer, 3), '---']
]

print(tabulate(table, headers = ['Metric', 'GZZ', 'With correction']))

if args.surf or args.scld:
	p = pv.Plotter()
	if args.surf:
		p.add_mesh(surf, color = True, show_edges = True)
	if args.scld:
		p.add_points(cloud, color = 'red', point_size = 3)
	p.show()

print(Footer)