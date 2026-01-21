__version__ = "0.1.0"


#things to be imported when "from blochK import *" is called
from .hamiltonian import Hamiltonian2D, BrillouinZone2D, Hamiltonian3D, BrillouinZone3D
from . import plotting
__all__ = ["Hamiltonian2D", "BrillouinZone2D","Hamiltonian3D", "BrillouinZone3D","plotting"]

#we want some of the utils to be directly accessible
from .utils import hamiltonian_fct
from .utils import parameters
