import numpy as np
from numpy import pi
from types import SimpleNamespace #for operators of Hamiltonian
import scipy


class Hamiltonian2D:
    def __init__(self, Hamiltonian_func, n1=np.array([1,0]),n2=np.array([0,1]), basis=None, basis_states=None,param={}):

        assert callable(Hamiltonian_func), "Hamiltonian_func must be a callable function"
        Hk = Hamiltonian_func(np.array([0]),np.array([0]))
        assert isinstance(Hk, np.ndarray), "Hamiltonian_func must return a numpy array"
        assert Hk.ndim == 3, "Hamiltonian_func must return a 3D array"
        assert Hk.shape[0] == Hk.shape[1], "Hamiltonian_func must return a square matrix for each k-point"
        assert Hk.shape[2] == 1, "Hamiltonian_func must return a 3D array with last dimensions equal to kx.shape"
        
        self.Hamiltonian_func = Hamiltonian_func  # the function defining H
        self.n_orbitals = Hamiltonian_func(np.array([0]),np.array([0]),**param).shape[0]  # number of orbitals (size of H)
        self.param = param  # parameters of Hamiltonian_func (e.g., hopping strengths)
        self.n1 = n1  # lattice vector 1
        self.n2 = n2  # lattice vector 2

        #Further consistency checks
        assert self.check_hermiticity(np.array([1.3,-0.5]),np.array([0.2,2.])), "Hamiltonian_func is not Hermitian"
        
        #define corresponding Brillouin zone
        n1_3D = np.array([n1[0], n1[1], 0])
        n2_3D = np.array([n2[0], n2[1], 0])
        zunit = np.array([0,0,1])
        self.BZ = BrillouinZone2D(m1=2*pi*np.cross(zunit,n2_3D)[:2]/np.vdot(n1_3D,np.cross(zunit,n2_3D)), m2=2*pi*np.cross(zunit,n1_3D)[:2]/np.vdot(n2_3D,np.cross(zunit,n1_3D)))  # Brillouin zone object

        # Define basis
        self.basis = None  # basis [spin, sublattice, orbital, ...]
        self.basis_states = None  # names of basis states ['up','down',...
        self.operator = SimpleNamespace()  # operators acting on basis states (e.g., spin and sublattice Pauli matrices)
        self.suboperator = SimpleNamespace()  # operators acting on part of basis states


    def set_params(self, kwargs):
        """set parameters of Hamiltonian_func"""
        self.param = kwargs


    def update_params(self, kwargs):
        """set parameters of Hamiltonian_func"""
        self.param.update(kwargs)

    
    def add_operator(self, name: str, operator: np.ndarray):
        """Add operator acting on basis states
        Parameters:
        name: str
            Name of the operator (e.g., 'spin', 'sublattice')
        operator: np.ndarray
            Operator matrix of shape (n_orbitals, n_orbitals) or (n_orbitals,)
        """
        if operator.shape != (self.n_orbitals, self.n_orbitals) and operator.shape != (self.n_orbitals,):
            raise ValueError(f"Operator must be of shape ({self.n_orbitals}, {self.n_orbitals}) or ({self.n_orbitals},)")
        self.operator.__setattr__(name, operator)


    def add_suboperator(self, name: str, operator: np.ndarray):
        """Add operator acting on part of basis states
        Parameters:
        name: str
            Name of the operator (e.g., 'spin', 'sublattice')
        operator: np.ndarray
            Operator matrix of shape (<n_orbitals, <n_orbitals) or (<n_orbitals,)
        """
        self.suboperator.__setattr__(name, operator)


    def evaluate(self, kx, ky, override_params={}):
        """Evaluate Hamiltonian at (kx,ky) with optional parameter overrides
        Parameters:
        kx, ky: np.ndarray
        override_params: dict
        Returns:
        Hk: np.ndarray of shape (n_orbitals, n_orbitals, *kx.shape)
        """

        # Merge object params and any overrides
        param = self.param.copy()
        param.update(override_params)
        return self.Hamiltonian_func(kx, ky, **param)
    

    def check_hermiticity(self, kx, ky):
        """Check if Hamiltonian_fct is Hermitian at (kx,ky)"""
        Hk = self.evaluate(kx, ky)  # shape (n_orbitals, n_orbitals, *kx.shape)
        Hk_dag = np.conjugate(np.swapaxes(Hk, 0, 1))  # Hermitian conjugate
        return np.allclose(Hk, Hk_dag)
    

    def __add__(self, other):
        """Add two Hamiltonians together. The resulting Hamiltonian_func is the sum of the two Hamiltonian_funcs. The parameters and operators are merged. If there are overlapping parameter or operator names, the ones from the self Hamiltonian will overwrite the other.
        Use with caution, as this can lead to unexpected results if the two Hamiltonians have different parameter or operator names that are not intended to be merged.
        """

        if not isinstance(other, Hamiltonian2D):
            raise ValueError("Can only add another Hamiltonian2D object")
        
        if self.n_orbitals != other.n_orbitals:
            raise ValueError("Incompatible Hamiltonian dimensions")

        # Define new Hamiltonian_func as sum of the two
        def new_Hamiltonian_func(kx, ky, **param):
            return self.evaluate(kx, ky, param) + other.evaluate(kx, ky, param)
        
        # Merge parameters and operators
        new_param = self.param.copy()
        new_param.update(other.param)
        
        new_hamiltonian = Hamiltonian2D(new_Hamiltonian_func, n1=self.n1, n2=self.n2, param=new_param, basis=self.basis, basis_states=self.basis_states)
        
        # Merge operators
        for name in set(self.operator.__dict__.keys()).union(set(other.operator.__dict__.keys())):
            if hasattr(self.operator, name) and hasattr(other.operator, name):
                # If both have the same operator name, we can choose to add them or overwrite. Here we choose to add.
                new_operator = self.operator.__dict__.get(name, 0) + other.operator.__dict__.get(name, 0)
            else:
                # If only one has the operator, take that one
                new_operator = self.operator.__dict__.get(name) or other.operator.__dict__.get(name)
            new_hamiltonian.add_operator(name, new_operator)

        # Merge suboperators in the same way
        for name in set(self.suboperator.__dict__.keys()).union(set(other.suboperator.__dict__.keys())):
            if hasattr(self.suboperator, name) and hasattr(other.suboperator, name):
                new_suboperator = self.suboperator.__dict__.get(name, 0) + other.suboperator.__dict__.get(name, 0)
            else:
                new_suboperator = self.suboperator.__dict__.get(name) or other.suboperator.__dict__.get(name)
            new_hamiltonian.add_suboperator(name, new_suboperator)

        return new_hamiltonian


    def diagonalize(self, kx, ky, override_params={}):
        """Diagonalize Hamiltonian at (kx,ky) with optional parameter overrides
        Parameters:
        kx, ky: np.ndarray
        override_params: dict
        Returns:
        es: np.ndarray of shape (bands=n_orbitals, *kx.shape)
            Eigenvalues at each k-point
        psis: np.ndarray of shape (bands=n_orbitals, *kx.shape,n_orbitals)
            Eigenvectors at each k-point
        """
        Hk = self.evaluate(kx, ky, override_params)  # shape (n_orbitals, n_orbitals, *kx.shape)
        Hk = np.moveaxis(np.moveaxis(Hk,1,-1),0,-2) #make it ...x n_orbitals x n_orbitals dimensional
        es,vs = np.linalg.eigh(Hk)
        es = np.moveaxis(es,-1,0) #.shape=band x kys (x kxs)
        psi = np.moveaxis(vs,-1,0) #.shape=band x kys (x kxs) x n_orbitals

        return es, psi

    


class BrillouinZone2D:
    def __init__(self, m1= np.array([2*np.pi,0]), m2: np.ndarray = np.array([0,2*np.pi])):
        self.m1 = m1
        self.m2 = m2
        self.set_points(dict())
        self.area = np.abs(m1[0]*m2[1] - m1[1]*m2[0])


    def sample(self,Lk:int,oversample_edge:bool=False):
        """Sample the Brillouin zone on a Lk x Lk grid
        Parameters:
        Lk: int
            Number of k-points along each direction
        oversample_edge: bool
            If True, sample 1 point more along each direction. The true BZ is then ks[:,1:-1,1:-1].
            Useful for observables which involve |u_n(k)> and |u_n(k+dk)> if the Hamiltonian is gauge dependent. 
        Returns:
        ks: np.ndarray of shape (2, Lk, Lk)
            Sampled k-points in the Brillouin zone
        """
        epsilon = 1/Lk #ensures that the edge is not sampled twice
        if oversample_edge:
            idxs =np.linspace(-1-epsilon,1+epsilon,Lk+2)/2
        else:
            idxs =np.linspace(-1+epsilon,1-epsilon,Lk)/2
        ks = np.meshgrid(idxs, idxs, indexing='ij')
        #i*m_1 + j*m_2
        ks = np.einsum('ij,ixy->jxy',np.array([self.m1, self.m2]),ks)
        return ks
    

    def set_points(self, additional_points: dict, include_default_points=True):
        """Set high-symmetry points in the Brillouin zone"""
        default_points = {
            r'\Gamma': np.array([0,0]),
            'X': self.m1/2,
            'Y': self.m2/2,
            '-X': -self.m1/2,
            '-Y': -self.m2/2,
            'M': (self.m1+self.m2)/4,
            '-M': -(self.m1+self.m2)/4,
            "M'": (self.m1-self.m2)/4,
            "-M'": -(self.m1-self.m2)/4,
            'R': (self.m1+self.m2)/2,
            '-R': -(self.m1+self.m2)/2,
            "R'": (self.m1-self.m2)/2,
            "-R'": -(self.m1-self.m2)/2,
        }
        if include_default_points:
            self.points = dict(**default_points, **additional_points)
        else:
            self.points = additional_points 


    def return_boundary(self, periodic=False,true_BZ=True):
        """
        Compute vertices of the 1st Brillouin zone from reciprocal lattice vectors.
        Parameters
        ----------
        periodic : bool, optional
            If True, the polygon is closed by repeating the first vertex at the end. Default is False.
        true_BZ : bool, optional
            If True, compute the Wigner-Seitz cell of the reciprocal lattice. If False, compute the span spanned by b1 and b2. Default is True.
        Returns
        -------
        vertices : ndarray, shape (N, 2)
            Ordered vertices of the Brillouin zone polygon.
        """

        if true_BZ: 
            # generate reciprocal lattice points around origin
            grid_range = range(-2, 3)   # 5x5 neighborhood
            points = [m*self.m1 + n*self.m2 for m in grid_range for n in grid_range]
            points = np.array(points)
            
            # Voronoi diagram
            vor = scipy.spatial.Voronoi(points)
            
            # index of the origin
            origin_index = np.argmin(np.linalg.norm(points, axis=1))
            
            # region around the origin
            region_index = vor.point_region[origin_index]
            region = vor.regions[region_index]
            
            # vertices of BZ polygon
            vertices = vor.vertices[region]
            
            # order them counterclockwise
            center = vertices.mean(axis=0)
            angles = np.arctan2(vertices[:,1]-center[1], vertices[:,0]-center[0])
            vertices = vertices[np.argsort(angles)]
        else:# span spanned by m1 and m2
            vertices = np.array([np.zeros_like(self.m1), self.m1, self.m1 + self.m2, self.m2]) - (self.m1 + self.m2)/2

        
        if periodic:
            # Ensure the polygon is closed by repeating the first vertex at the end
            vertices = np.vstack([vertices, vertices[0]])
            
        return vertices




class Hamiltonian3D:
    def __init__(self, Hamiltonian_func, n1=np.array([1,0,0]),n2=np.array([0,1,0]),n3=np.array([0,0,1]), basis=None, basis_states=None,param={}):

        assert callable(Hamiltonian_func), "Hamiltonian_func must be a callable function"
        Hk = Hamiltonian_func(np.array([0]),np.array([0]),np.array([0]))
        assert isinstance(Hk, np.ndarray), "Hamiltonian_func must return a numpy array"
        assert Hk.ndim == 3, "Hamiltonian_func must return a 3D array"
        assert Hk.shape[0] == Hk.shape[1], "Hamiltonian_func must return a square matrix for each k-point"
        assert Hk.shape[2] == 1, "Hamiltonian_func must return a 3D array with last dimensions equal to kx.shape"
        
        self.Hamiltonian_func = Hamiltonian_func  # the function defining H
        self.n_orbitals = Hamiltonian_func(np.array([0]),np.array([0]),np.array([0]),**param).shape[0]  # number of orbitals (size of H)
        self.param = param  # parameters of Hamiltonian_func (e.g., hopping strengths)
        self.n1 = n1  # lattice vector 1
        self.n2 = n2  # lattice vector 2
        self.n3 = n3  # lattice vector 3

        #Further consistency checks
        assert self.check_hermiticity(np.array([1.3,-0.5]),np.array([0.2,2.]),np.array([0.1,0.3])), "Hamiltonian_func is not Hermitian"
        
        #define corresponding Brillouin zone
        volume = np.vdot(n1, np.cross(n2, n3))  # scalar triple product
        m1 = 2 * np.pi * np.cross(n2, n3) / volume
        m2 = 2 * np.pi * np.cross(n3, n1) / volume
        m3 = 2 * np.pi * np.cross(n1, n2) / volume
        self.BZ = BrillouinZone3D(m1=m1, m2=m2, m3=m3)

        # Define basis
        self.basis = None  # basis [spin, sublattice, orbital, ...]
        self.basis_states = None  # names of basis states ['up','down',...
        self.operator = SimpleNamespace()  # operators acting on basis states (e.g., spin and sublattice Pauli matrices)
        self.suboperator = SimpleNamespace()  # operators acting on part of basis states


    def set_params(self, kwargs):
        """set parameters of Hamiltonian_func"""
        self.param = kwargs


    def update_params(self, kwargs):
        """set parameters of Hamiltonian_func"""
        self.param.update(kwargs)

    
    def add_operator(self, name: str, operator: np.ndarray):
        """Add operator acting on basis states
        Parameters:
        name: str
            Name of the operator (e.g., 'spin', 'sublattice')
        operator: np.ndarray
            Operator matrix of shape (n_orbitals, n_orbitals) or (n_orbitals,)
        """
        if operator.shape != (self.n_orbitals, self.n_orbitals) and operator.shape != (self.n_orbitals,):
            raise ValueError(f"Operator must be of shape ({self.n_orbitals}, {self.n_orbitals}) or ({self.n_orbitals},)")
        self.operator.__setattr__(name, operator)


    def add_suboperator(self, name: str, operator: np.ndarray):
        """Add operator acting on part of basis states
        Parameters:
        name: str
            Name of the operator (e.g., 'spin', 'sublattice')
        operator: np.ndarray
            Operator matrix of shape (<n_orbitals, <n_orbitals) or (<n_orbitals,)
        """
        self.suboperator.__setattr__(name, operator)


    def evaluate(self, kx, ky, kz, override_params={}):
        """Evaluate Hamiltonian at (kx,ky) with optional parameter overrides
        Parameters:
        kx, ky: np.ndarray
        override_params: dict
        Returns:
        Hk: np.ndarray of shape (n_orbitals, n_orbitals, *kx.shape)
        """

        # Merge object params and any overrides
        param = self.param.copy()
        param.update(override_params)
        return self.Hamiltonian_func(kx, ky, kz, **param)
    

    def check_hermiticity(self, kx, ky, kz):
        """Check if Hamiltonian_fct is Hermitian at (kx,ky)"""
        Hk = self.evaluate(kx, ky, kz)  # shape (n_orbitals, n_orbitals, *kx.shape)
        Hk_dag = np.conjugate(np.swapaxes(Hk, 0, 1))  # Hermitian conjugate
        return np.allclose(Hk, Hk_dag)


    def diagonalize(self, kx, ky, kz, override_params={}):
        """Diagonalize Hamiltonian at (kx,ky) with optional parameter overrides
        Parameters:
        kx, ky: np.ndarray
        override_params: dict
        Returns:
        es: np.ndarray of shape (bands=n_orbitals, *kx.shape)
            Eigenvalues at each k-point
        psis: np.ndarray of shape (bands=n_orbitals, *kx.shape,n_orbitals)
            Eigenvectors at each k-point
        """
        Hk = self.evaluate(kx, ky, kz, override_params)  # shape (n_orbitals, n_orbitals, *kx.shape)
        Hk = np.moveaxis(np.moveaxis(Hk,1,-1),0,-2) #make it ...x n_orbitals x n_orbitals dimensional
        es,vs = np.linalg.eigh(Hk)
        es = np.moveaxis(es,-1,0) #.shape=band x kys (x kxs)
        psi = np.moveaxis(vs,-1,0) #.shape=band x kys (x kxs) x n_orbitals

        return es, psi

    


class BrillouinZone3D:
    def __init__(self, m1= np.array([2*np.pi,0,0]), m2: np.ndarray = np.array([0,2*np.pi,0]), m3: np.ndarray = np.array([0,0,2*np.pi])):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.set_points(dict())
        self.volume = np.abs(np.dot(m1, np.cross(m2, m3)))


    def sample(self,Lk:int,oversample_edge:bool=False):
        """Sample the Brillouin zone on a Lk x Lk x Lk grid
        Parameters:
        Lk: int
            Number of k-points along each direction
        oversample_edge: bool
            If True, sample 1 point more along each direction. The true BZ is then ks[:,1:-1,1:-1].
            Useful for observables which involve |u_n(k)> and |u_n(k+dk)> if the Hamiltonian is gauge dependent. 
        Returns:
        ks: np.ndarray of shape (3, Lk, Lk, Lk)
            Sampled k-points in the Brillouin zone
        """
        epsilon = 1/Lk #ensures that the edge is not sampled twice
        if oversample_edge:
            idxs =np.linspace(-1-epsilon,1+epsilon,Lk+2)/2
        else:
            idxs =np.linspace(-1+epsilon,1-epsilon,Lk)/2
        ks = np.meshgrid(idxs, idxs, idxs, indexing='ij')
        #i*m_1 + j*m_2 + k*m_3
        ks = np.einsum('ij,ixyz->jxyz',np.array([self.m1, self.m2, self.m3]),ks)
        return ks
    

    def set_points(self, additional_points: dict, include_default_points=True):
        """Set high-symmetry points in the Brillouin zone"""
        default_points = {
            r'\Gamma': np.array([0, 0, 0]),
            # Face centers
            'X':  self.m1 / 2,
            'Y':  self.m2 / 2,
            'Z':  self.m3 / 2,
            '-X': -self.m1 / 2,
            '-Y': -self.m2 / 2,
            '-Z': -self.m3 / 2,
            # Edge centers
            'M':  (self.m1 + self.m2) / 2,
            'N':  (self.m1 + self.m3) / 2,
            'P':  (self.m2 + self.m3) / 2,
            '-M': -(self.m1 + self.m2) / 2,
            '-N': -(self.m1 + self.m3) / 2,
            '-P': -(self.m2 + self.m3) / 2,
            # Body corner
            'R':  (self.m1 + self.m2 + self.m3) / 2,
            '-R': -(self.m1 + self.m2 + self.m3) / 2,
        }
        if include_default_points:
            self.points = dict(**default_points, **additional_points)
        else:
            self.points = additional_points 


    def return_boundary(self, grid_range=2):
        """
        Compute vertices and faces of the 3D Brillouin zone (Wigner-Seitz cell
        of the reciprocal lattice) from reciprocal lattice vectors m1, m2, m3.

        Parameters
        ----------
        grid_range : int, optional
            Lattice points are generated for indices in range(-grid_range, grid_range+1)
            along each of the 3 reciprocal directions. Increase this if you get a
            warning about unbounded ridges (very oblique/skewed lattices need more).

        Returns
        -------
        vertices : ndarray, shape (N, 3)
            Vertices of the BZ polyhedron.
        faces : list of list of int
            Each entry is a list of vertex indices (into `vertices`), ordered
            counterclockwise around that face (as seen from outside).
        """
        rng = range(-grid_range, grid_range + 1)
        points = np.array([m*self.m1 + n*self.m2 + p*self.m3
                            for m in rng for n in rng for p in rng])

        vor = scipy.spatial.Voronoi(points)

        origin_index = np.argmin(np.linalg.norm(points, axis=1))

        # collect ridges (faces) that border the origin's cell
        raw_faces = []
        used_global_ids = set()
        for (p1, p2), ridge_verts in zip(vor.ridge_points, vor.ridge_vertices):
            if origin_index == p1 or origin_index == p2:
                if -1 in ridge_verts:
                    raise RuntimeError(
                        "Unbounded ridge encountered - increase grid_range."
                    )
                raw_faces.append(ridge_verts)
                used_global_ids.update(ridge_verts)

        # build compact local vertex list
        global_ids_sorted = sorted(used_global_ids)
        global_to_local = {g: i for i, g in enumerate(global_ids_sorted)}
        vertices = vor.vertices[global_ids_sorted]

        # order vertices of each face counterclockwise (planar polygon sort)
        faces = []
        for ridge_verts in raw_faces:
            pts_global = np.array(ridge_verts)
            pts = vor.vertices[pts_global]
            center = pts.mean(axis=0)

            normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            normal /= np.linalg.norm(normal)

            u = pts[0] - center
            u /= np.linalg.norm(u)
            v = np.cross(normal, u)

            d = pts - center
            angles = np.arctan2(d @ v, d @ u)
            order = np.argsort(angles)

            local_ids = [global_to_local[g] for g in pts_global[order]]
            faces.append(local_ids)

        return vertices, faces