from blochK.plotting.plotting import plot_FS,plot_bandstruc
from blochK.hamiltonian_testing import create_Hsquare
from matplotlib import pyplot as plt


def test_plot_FS():
    Hsquare = create_Hsquare()
    Hsquare.set_params(dict(m=0.5)) #set parameters of Hamiltonian

    fig,axs = plt.subplots(1,2, figsize=(6,3))

    axs[0].set_title('Fermi Surface uncolored')
    plot_FS(axs[0], Hsquare, Lk=5, coloring_operator='k')

    axs[1].set_title('Fermi Surface colored by spin')
    plot_FS(axs[1], Hsquare, Lk=5, coloring_operator=Hsquare.operator.spin,cmap='bwr',show_ylabel=False)


def test_plot_bandstruc():
    Hsquare = create_Hsquare()
    Hsquare.set_params(dict(m=0.5)) #set parameters of Hamiltonian

    fig,axs = plt.subplots(1,2, figsize=(6,3))

    #define a path in the BZ
    labels_points_path = ['\Gamma','X','R','Y','\Gamma'] 
    #the right poitns are automatically found from the BZ points

    fig,axs = plt.subplots(1,2, figsize=(6,3))

    axs[0].set_title('Fermi Surface uncolored')
    plot_bandstruc(axs[0], Hsquare, N_samples=5, coloring_operator='k',labels_points_path=labels_points_path)

    axs[1].set_title('Fermi Surface colored by spin')
    plot_bandstruc(axs[1], Hsquare, N_samples=5,labels_points_path=labels_points_path, coloring_operator=Hsquare.operator.spin,cmap='bwr',show_ylabel=False)