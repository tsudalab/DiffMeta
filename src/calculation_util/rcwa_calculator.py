import os
import S4
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from numpy import array
import numba as nb
from scipy.io import loadmat
from scipy import interpolate
import time
from PIL import Image
def materialepsilon(wavelength_um, data):
    wav =data[:,0]
    n0 = data[:,1]
    k0 = data[:,2]
    mn1 = interpolate.interp1d(wav, n0,fill_value='extrapolate')
    na=mn1(wavelength_um)
    if na<0:
        na=10**(-20)

    mk1 = interpolate.interp1d(wav, k0,fill_value='extrapolate')
    ka=mk1(wavelength_um)
    if ka<0:
        ka=10**(-20)
    
    return na,ka


def maincc(ind):
    lam = wavelength_space[ind]
    f = 1 / float(lam)
    S.SetFrequency(f)

    na_aSi,ka_aSi =materialepsilon(lam,mat_aSi)
    na_ZnS,ka_ZnS =materialepsilon(lam,mat_ZnS)
    na_SiO2,ka_SiO2 =1.5,0
    na_ZnS,ka_ZnS=3.42,0
    S.SetMaterial(Name = 'aSi',Epsilon = (complex(na_aSi,ka_aSi))**2)
    S.SetMaterial(Name = 'ZnS',Epsilon = (complex(na_ZnS,ka_ZnS))**2)
    S.SetMaterial(Name = 'SiO2',Epsilon = (complex(na_SiO2,ka_SiO2))**2)

    (forw,back) = S.GetPowerFlux(Layer = 'AirAbove', zOffset = 0)
    (forw_b,back_b) = S.GetPowerFlux(Layer = 'AirBelow', zOffset = 0)

    Refi = np.abs(back)
    Trani = np.abs(forw_b)

    return Refi,Trani

def run_simulation(polarization):
    if polarization == 's':
        S.SetExcitationPlanewave(
            IncidenceAngles=(theta, 0),
            sAmplitude = 1,
            pAmplitude = 0,
            Order = 0
        )
    elif polarization == 'p':
        S.SetExcitationPlanewave(
            IncidenceAngles=(theta, 0),
            sAmplitude = 0,
            pAmplitude = 1,
            Order = 0
        )
    else:
        raise ValueError("Invalid polarization: " + polarization)

    pool = multiprocessing.Pool(processes = 20)
    ss,sss=zip(*array(pool.map(maincc, range(nl))))
    ss=np.asarray(ss)
    sss=np.asarray(sss)
    pool.close()

    return np.real(ss.T), np.real(sss.T)


# The .mat file is loaded as a dictionary. 
# You can access the 'all_imags' variable like this:
mat = loadmat('../data/all_images_new.mat')
all_imags = mat['all_images']
all_imags = np.squeeze(all_imags, axis=2) 

Si_data_exp=np.loadtxt('Palik_Au.txt')
mat_aSi=np.array(Si_data_exp)

ZnS_data_exp=np.loadtxt('Palik_ZnS.txt')
mat_ZnS=np.array(ZnS_data_exp)

data = np.load('../../data/Dataset_non_pol_results_Si.npz')
all_p = data['p'] 
all_slab_thickness = data['slab_thickness']  
all_space_thickness = data['space_thickness'] 
all_bottom_thickness = data['bottom_thickness']

npz=0
start_ind=10000*(npz)
end_ind=10000*(npz+1)
test_id = 99

# Loop over all images
for img_index in range(start_ind, end_ind, test_id):
    img = all_imags[img_index]

    start_time=time.time()
    a = 1e-6; #1nm
    c_const = 3e8;

    # p = np.random.uniform(3, 8)  # periodic length in um
    # slab_thickness = np.random.uniform(0, 0.8)  # slab layer thickness in um
    # space_thickness = np.random.uniform(0, 1)  # space layer thickness in um
    # bottom_thickness = np.random.uniform(0, 0.2)  # space layer thickness in um
    p, slab_thickness, space_thickness, bottom_thickness = all_p[img_index],all_slab_thickness[img_index],all_space_thickness[img_index],all_bottom_thickness[img_index]
    period = [p,p]

    Num_ord=40

    S = S4.New(Lattice=((period[0],0),(0,period[1])),NumBasis=Num_ord)

    S.SetMaterial(Name = 'aSi',Epsilon = (1 + 0j)**2)
    S.SetMaterial(Name = 'ZnS',Epsilon = (1 + 0j)**2)
    S.SetMaterial(Name = 'SiO2',Epsilon = (1 + 0j)**2)
    S.SetMaterial(Name = 'Vacuum',Epsilon = (1 + 0j)**2)

    S.AddLayer(Name = 'AirAbove',Thickness = 0, Material = 'Vacuum')
    S.AddLayer(Name = 'slab',Thickness = slab_thickness, Material = 'Vacuum')
   
    width, height = img.shape

    # Define your materials
    n_air = 'Vacuum'
    n_medium = 'aSi'
    pixel_unit = p/64  # adjust this according to your needs

    # Calculate the shifts
    width_shift = width / 2
    height_shift = height / 2

    # Loop over the pixels in the image
    for i in range(width):
        for j in range(height):
            # Calculate the coordinates in the S4 simulation
            x = (i - width_shift + 0.5) * pixel_unit
            y = (j - height_shift + 0.5) * pixel_unit

            # Check the value of the pixel in the image
            if img[i, j] <= 0.5:
                # If the pixel value is less than or equal to 0.5, set the material to n_air
                material = n_air
            else:
                # Otherwise, set the material to n_medium
                material = n_medium

            # Set the region in the S4 simulation
            S.SetRegionRectangle(
                Layer='slab',
                Material=material,
                Center=(x, y),
                Angle=0,
                Halfwidths=(pixel_unit / 2, pixel_unit / 2)
            )    

    S.AddLayer(Name = 'space',Thickness = space_thickness,Material = 'ZnS')      
    S.AddLayer(Name = 'bottom',Thickness = bottom_thickness,Material = 'aSi')
    S.AddLayer(Name = 'AirBelow',Thickness = 0.0,Material = 'SiO2')

    
    theta=0

    # frequency sweep
    nl=400
    wavelength_space = np.linspace(3,15,nl)

    # Run the simulation for s-polarized light
    Ref_s, Tran_s = run_simulation('s')


    # Run the simulation for p-polarized light
    Ref_p, Tran_p = run_simulation('p')

    # Average the results
    Ref = (Ref_s + Ref_p) / 2
    Tran = (Tran_s + Tran_p) / 2
    kesi = 1 - Ref - Tran
    plt.plot(wavelength_space,kesi,label=f'Generated_{(img_index)}')
    
plt.legend()
plt.savefig('spectrum.png')