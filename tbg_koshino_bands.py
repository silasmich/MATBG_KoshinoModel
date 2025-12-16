import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

m, n = 32, 31    # 1.05º (magic-angle)
HOPPING_FILE = "eff_hopping_ver2.dat"

# Read the hopping file (numeric table)
rows = []
with open(HOPPING_FILE, 'r') as f:
    for line in f:
        s = line.strip()
        parts = s.split()
        vals = [float(x) for x in parts]
        rows.append(vals)

# Interpret columns as (m,n,ia,jb,Re,Im)
arr = np.array(rows)
m_vals = arr[:,0].astype(int)    # lattice vector index
n_vals = arr[:,1].astype(int)    # lattice vector index
ia = arr[:,2].astype(int)        # initial orbital
jb = arr[:,3].astype(int)        # final orbital
re, im = arr[:,4], arr[:,5]
tvals = re + 1j*im               # ξ=+ hopping integrals

# Aggregate hoppings into a dictionary keyed by (i,j,m,n)
hops = {}
for mv,nv,i0,j0,t in zip(m_vals, n_vals, ia, jb, tvals):
    i = int(i0) - 1   # convert 1-based -> 0-based
    j = int(j0) - 1
    key = (i,j,int(mv),int(nv))
    hops[key] = np.conj(t)
print("Loaded hopping entries:", len(hops))

# Graphene reciprocal lattice vectors 
def graphene_reciprocal():
    a = sqrt(3.0)*1.42
    a1 = np.array([a, 0.0])
    a2 = np.array([a/2.0, a*sqrt(3.0)/2.0])
    A = np.vstack((a1, a2))
    B = 2*np.pi*np.linalg.inv(A.T)
    return B[0,:], B[1,:]

# Twist angle. Given by doi:10.1103/PhysRevB.81.161405
def commensurate_angle():
    num = m*m + 4*m*n + n*n
    den = 2*(m*m + m*n + n*n)
    cos_th = np.clip(num/den, -1.0, 1.0)
    return np.arccos(cos_th)    # in radians

# Rotation matrix
def R(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

# Moiré reciprocal and real-space lattice vectors
def moire_vectors():
    theta = commensurate_angle()
    b1, b2 = graphene_reciprocal()
    g1 = R(-theta/2) @ b1 - R(+theta/2) @ b1
    g2 = R(-theta/2) @ b2 - R(+theta/2) @ b2
    G = np.vstack((g1, g2))
    S = 2*np.pi*np.linalg.inv(G.T)
    return G, S, theta

G, S, theta = moire_vectors()
print(f"Angle = {np.degrees(theta):.6f}°; (m,n)=({m},{n})")
print("Moiré reciprocal vectors (1/Å):\n", G)
print("Moiré lattice vectors (Å):\n", S)

# Wannier centers (2 per moiré cell)
def centers():
    a = sqrt(3.0)*1.42
    Lm = a/(2*np.sin(theta/2))
    rBA = np.array([Lm/(2*np.sqrt(3)), Lm/2])
    rAB = np.array([-Lm/(2*np.sqrt(3)), Lm/2])
    return np.vstack((rBA, rAB))
print("Wannier centers (2):\n", centers())

# High-symmetry points. Negative signs match Koshino's article
g1 = G[:,0]; g2 = G[:,1]
Gamma = np.array([0.0,0.0])
K = -(2*g1 + g2)/3.0
M = -(g1 + g2)/2.0

# Path in mBZ proportional to distances
segpts = 150
r_segpts = int(segpts/np.sqrt(3))
seg1 = np.linspace(K, Gamma, r_segpts*2, endpoint=False)
seg2 = np.linspace(Gamma, M, segpts, endpoint=False)
seg3 = np.linspace(M, K, r_segpts+1)
kpts = np.vstack((seg1, seg2, seg3))
size_kpts = len(kpts)

# Build ξ-valley Hamiltonian at k
def H_of_k(kvec, valley):
    H = np.zeros((2,2), dtype=complex)
    if (valley==+1):
        for (i,j,mv,nv), t in hops.items():
            Rcart = mv * S[:,0] + nv * S[:,1]
            phase = np.exp(1j * (kvec[0]*Rcart[0] + kvec[1]*Rcart[1]))
            H[i,j] += t * phase
    if (valley==-1):
        for (i,j,mv,nv), t in hops.items():
            Rcart = mv * S[:,0] + nv * S[:,1]
            phase = np.exp(1j * (kvec[0]*Rcart[0] + kvec[1]*Rcart[1]))
            H[i,j] += np.conj(t) * phase
    return H

# Compute the ξ-valley band structure in meV
def bandstr(valley):
    bands = np.zeros((size_kpts, 2))
    for ik,k in enumerate(kpts):
        Hk = H_of_k(k, valley)
        vals = np.linalg.eigvalsh(Hk)
        bands[ik,:] = np.sort(np.real(vals))*1000
    return bands

# Returns the closest index to a high-symmetry point
def closest_index(pt):
    d = np.linalg.norm(kpts-pt, axis=1)
    return int(np.argmin(d))

# Print energies at high-symmetry points
def E(bands):
    for name,pt in [('Gamma',Gamma),('K',K),('M',M)]:
        idx = closest_index(pt)
        print(f"{name} energies: {bands[idx]}")

# Band structure layout
def layout():
    plt.figure(figsize=(8,5))
    plt.title(f"TBG effective bands (Koshino) for 1.05°")
    plt.ylabel("Energy (meV)")
    plt.ylim(top=4, bottom=-4.9)
    plt.margins(x=0)    # bands touch axes
        
    # Add labels for high-symmetry points
    plt.xticks(ticks=[0, closest_index(Gamma), closest_index(M), size_kpts-1],
               labels=[r'$\overline{K}$', r'$\overline{\Gamma}$', r'$\overline{M}$', r"$\overline{K}'$"], fontsize=12)

    # Add lines
    plt.axhline(y=0, color='black', ls='--', linewidth=0.8, alpha=1.0)
    plt.axvline(closest_index(Gamma), color='black', ls='-', linewidth=0.8, alpha=1.0)
    plt.axvline(closest_index(M), color='black', ls='-', linewidth=0.8, alpha=1.0)
    plt.tick_params(length=0)
    plt.tight_layout()    

def main():
    bands_p, bands_m = bandstr(+1), bandstr(-1)    # valleys ξ = ±
    k_path = np.arange(size_kpts)
    
    # Plot bands
    layout()
    plt.plot(k_path, bands_p[:,0], color='orangered', label=r'$\xi=+$')
    plt.plot(k_path, bands_p[:,1], color='orangered')
    plt.plot(k_path, bands_m[:,0], color='royalblue', label=r'$\xi=-$')
    plt.plot(k_path, bands_m[:,1], color='royalblue')
    plt.legend()
    
    plt.savefig("Koshino_Bands", dpi=600)
    plt.show()
    # E(bands_p)
    # E(bands_m)

if __name__ == "__main__":
    main()
