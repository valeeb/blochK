import numpy as np

####################################################################
# Matplotlib settings for publication style figures.
####################################################################

# Panels at least one column wide
paper_style  = {
        ## LaTeX fonts is turned off, because its readability is worse
        #"text.usetex": True,
        #"font.family": "serif",
        "axes.titlesize": 9,
        "axes.labelsize": 8, #actually 10
        "font.size": 8, #10
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        'lines.markersize':4, #half the usual value
        'lines.markeredgewidth': 0.7, #1
        'lines.linewidth':0.7, 
        'axes.linewidth':0.5, #half the usual
        'xtick.major.width':0.4, #half the usual
        'ytick.major.width':0.4, #half the usual
        'xtick.top':True,
        'xtick.bottom':True,
        'ytick.left':True,
        'ytick.right':True,
        'xtick.direction':'in',
        'ytick.direction':'in'
    }

# Panels less than one column wide
paper_style_small  = {
        ## LaTeX fonts is turned off, because its readability is worse
        #"text.usetex": True,
        #"font.family": "serif",
        "axes.titlesize": 8,
        "axes.labelsize": 6, #actually 10
        "font.size": 6, #10
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        'lines.markersize':3, #half the usual value
        'lines.markeredgewidth': 0.3, #1
        'lines.linewidth':0.6,
        'axes.linewidth':0.4, #half the usual
        'xtick.major.width':0.3, 
        'ytick.major.width':0.3, 
        'xtick.major.size':2,
        'ytick.major.size':2,
        'xtick.top':True,
        'xtick.bottom':True,
        'ytick.left':True,
        'ytick.right':True,
        'xtick.direction':'in',
        'ytick.direction':'in'
    }

textwidth = 418.26
revtex_textwidth = 504.0 # width for a wide figure in revtex
revtex_columnwidth = 246.0 # width for a standard figure in revtex


##############################################################
# Function to set figure dimensions to avoid scaling in LaTeX
##############################################################

def set_size(width, fraction=1, ratio=(5**.5 - 1)/2, subplots=(1, 1),pad=0,width_ratio=1,extraheight=0):
    """Set figure dimensions to avoid scaling in LaTeX.
    Parameters
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which the figure to occupies
    ratio: float, optional
            Fraction of the width which the figure is high   
    subplots: array-like, optional
            The number of rows and columns of subplots
    pad: float, optional
            pad that is entered when saving the figure
    -------
    Returns
    fig_dim: tuple
            Dimensions of figure in inches
    """
    fig_width_pt = width * fraction
    inches_per_pt = 1/72.27 # Convert from pt to inches
    fig_width_in = fig_width_pt * inches_per_pt # Figure width in inches
    fig_height_in = fig_width_in*ratio*(subplots[0]/subplots[1])*width_ratio+extraheight*inches_per_pt # Figure height in inches
    fig_dim = (fig_width_in+2*(0.1-pad), fig_height_in+2*(0.1-pad))
    return fig_dim


def roundErr(x,dx):
    """rounds number x and its error dx such that dx has only one non-zero decimal"""
    po10 = np.logspace(-10, 10, 21)
    j = 10-po10.searchsorted(dx)+1 #number of decimalpoints to which should be rounded
    dx = np.round(dx,j)
    x = np.round(x,j)
    return x, dx


