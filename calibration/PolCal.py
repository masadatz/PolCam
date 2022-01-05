
# This code is for polarimetric calibration, after correction for in-homogeneity.

import cv2
import polanalyser as pa
from PIL import Image
import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt
from mpldatacursor import datacursor
import copy
import RadCal

#Simulate groud truth normalized Stokes Vector
def sim_GT_Snorm(AoLP_true, DoLP_true = np.array([0.999]), haxis = 2048, vaxis = 2448):
    if AoLP_true.shape == ():
        AoLP_true = np.array([AoLP_true])
    if DoLP_true.shape == ():
        DoLP_true = np.array([DoLP_true])
    n_im= AoLP_true.shape[0]
    GT_Snorm = np.ones([3,n_im,haxis,vaxis])
    q_true_val = DoLP_true*np.cos(2*AoLP_true)
    u_true_val = DoLP_true*np.sin(2*AoLP_true)
    for i in range(n_im):
        GT_Snorm[1,i,:,:] = q_true_val[i]
        GT_Snorm[2,i,:,:] = u_true_val[i]
    return GT_Snorm

#Simulate images, no demosaicing
# Two types of error, 1. Random noise, averaged over av_n images
#                     2. Error per pixel type, taken from normal STD
def sim_Noisy_images(GT_Snorm, error_by_type, av_n=5):
    t_guess = 1
    alpha_d = np.ones(GT_Snorm[0].shape)
    L_noise = np.ones(GT_Snorm[0].shape)
    alpha_d[:,1::2, 1::2]=  0
    alpha_d[:,0::2, 1::2]=  45
    alpha_d[:,0::2, 0::2]=  90
    alpha_d[:,1::2, 0::2]=  135
    alpha = np.deg2rad(alpha_d)
    L = 0.5*t_guess*(GT_Snorm[0]+ GT_Snorm[1]*np.cos(2*alpha)+GT_Snorm[2]*np.sin(2*alpha))
    #av_n simulates the number of photos averaged for single Stokes vector image
    noise =  np.mean(np.random.normal(0.0,1/100,np.array([av_n,L.shape[0],L.shape[1],L.shape[2]])),axis=0)
    L_noise[:, 1::2, 1::2] = L[:, 1::2, 1::2] * (1 + error_by_type[0])
    L_noise[:, 0::2, 1::2] = L[:, 0::2, 1::2] * (1 + error_by_type[1])
    L_noise[:, 0::2, 0::2] = L[:, 0::2, 0::2] * (1 + error_by_type[2])
    L_noise[:, 1::2, 0::2] = L[:, 1::2, 0::2] * (1 + error_by_type[3])
    L_noise = L_noise*(1+noise)
    return L_noise

#Show noisy images
def save_sim_images(images,show = False):
    for i in range(images.shape[0]):
        plt.figure()
        im1 = plt.imshow(images[i,:,:])
        im = Image.fromarray(images[i,:,:])
        im.save(f'{i}.tiff')
        plt.colorbar(im1)
        datacursor()
        plt.title('I')
        plt.axis('off')
        if show:
           plt.show()

# DoLP and AoLP error
def mean_error( AoLP_true, images, GT_Snorm, X_mat, DoLP_true = 0.999,
               calibrate = False):
    n_im = images.shape[0]
    vaxis = images.shape[1]
    haxis = images.shape[2]
    DoLP_Error = np.zeros(n_im)
    AoLP_Error = np.zeros(n_im)
    q_Error = np.zeros(n_im)
    u_Error = np.zeros(n_im)
    AoLP_sum = 0
    for i in range(n_im):
        images_demosaiced = pa.demosaicing(images[i])
        img_0, img_45, img_90, img_135 = cv2.split(images_demosaiced)
        # for real images we need normalization

        Stokes = pa.calcLinearStokes(np.moveaxis(np.array([img_0, img_45, img_90, img_135]), 0, -1), np.deg2rad([0,45,90,135]))
        if calibrate:
            Stokes_cal = Cal(Stokes, X_mat)
            Stokes = Stokes_cal
        DoLP_with_noise= pa.cvtStokesToDoLP(Stokes)
        AoLP_with_noise = pa.cvtStokesToAoLP(Stokes)
        DoLP_Error[i] = np.sum(np.abs(DoLP_with_noise-DoLP_true))/(haxis*vaxis)
        AoLP_Error[i] = np.sum(np.abs(AoLP_with_noise-AoLP_true[i]))/(haxis*vaxis)

        q_Error[i] = np.sum(np.abs(Stokes[...,1]-GT_Snorm[1,i,:,:]))
        u_Error[i] = np.sum(np.abs(Stokes[...,2]-GT_Snorm[2,i,:,:]))

    return [DoLP_Error, AoLP_Error]

def Cal_params(images, GT_Snorm):
    #calibration

    n_im = images.shape[0]
    vaxis = images.shape[2]
    haxis = images.shape[1]

    B_Mat = np.zeros([n_im,haxis*vaxis])
    A_Mat = np.zeros([n_im, 3])
    q_true = GT_Snorm[1,:,1,1]
    u_true = GT_Snorm[2,:,1,1]

    for i in range(n_im):
        B_Mat[i] = images[i].flatten()

    A_Mat[:,0]=1
    A_Mat[:,1] = np.transpose(q_true)
    A_Mat[:,2] = np.transpose(u_true)


    x = np.linalg.lstsq(A_Mat, B_Mat,rcond = None)
    X_mat = np.reshape(np.array(x[0]),[3,haxis,vaxis])# each column has the three parameters for a pixel
    return X_mat

def Cal(Stokes, X_mat):
    Stokes_cal = np.ones(Stokes.shape)
    q_meas = Stokes[...,1]
    u_meas = Stokes[...,2]
    H = q_meas.shape[0]
    V = q_meas.shape[1]
    a1_in = pa.demosaicing(X_mat[0])
    a2_in = pa.demosaicing(X_mat[1])
    a3_in = pa.demosaicing(X_mat[2])
    a1, c1, b1, d1 = cv2.split(a1_in)
    a2, c2, b2, d2 = cv2.split(a2_in)
    a3, c3, b3, d3 = cv2.split(a3_in)
    ab1 = a1-b1
    cd1 = c1-d1
    ab2 = a2-b2
    cd2 = c2-d2
    ab3 = a3-b3
    cd3 = c3-d3
    for i in range(H):
        for j in range(V):
            M = np.array([[ab2[i,j], ab3[i,j]],[cd2[i,j], cd3[i,j]]])
            #Min = np.linalg.inv(M)
            Min = inv(M)
            S =  np.array([  [ q_meas[i,j]-ab1[i,j] ],
                             [ u_meas[i,j]-cd1[i,j] ]  ])
            Scal = np.matmul(Min, S)
            Stokes_cal[i,j,1]=Scal[0]
            Stokes_cal[i,j,2]=Scal[1]
    return  Stokes_cal


def Cal_params_after_demo(images, GT_Snorm):
    #calibration

    n_im = images.shape[0]
    vaxis = images.shape[2]
    haxis = images.shape[1]

    B0_Mat = np.zeros([n_im, haxis * vaxis])
    B45_Mat = np.zeros([n_im, haxis * vaxis])
    B90_Mat = np.zeros([n_im, haxis * vaxis])
    B135_Mat = np.zeros([n_im, haxis * vaxis])

    A_Mat = np.zeros([n_im, 3])
    q_true = GT_Snorm[1,:,1,1]
    u_true = GT_Snorm[2,:,1,1]
    A_Mat[:,0]=1
    A_Mat[:,1] = np.transpose(q_true)
    A_Mat[:,2] = np.transpose(u_true)


    for i in range(n_im):
        images_demosaiced = pa.demosaicing(images[i])
        img_0, img_45, img_90, img_135 = cv2.split(images_demosaiced)
        B0_Mat[i] = img_0.flatten()
        B45_Mat[i] = img_45.flatten()
        B90_Mat[i] = img_90.flatten()
        B135_Mat[i] = img_135.flatten()

    Xa = np.linalg.lstsq(A_Mat, B0_Mat, rcond = None)
    Xc = np.linalg.lstsq(A_Mat, B45_Mat, rcond= None)
    Xb = np.linalg.lstsq(A_Mat, B90_Mat, rcond = None)
    Xd = np.linalg.lstsq(A_Mat, B135_Mat, rcond= None)

    a = np.reshape(np.array(Xa[0]), [3, haxis, vaxis])
    b = np.reshape(np.array(Xb[0]), [3, haxis, vaxis])
    c = np.reshape(np.array(Xc[0]), [3, haxis, vaxis])
    d = np.reshape(np.array(Xd[0]), [3, haxis, vaxis])
    # each has the three parameters for a pixel
    return a,b,c,d

def Cal_after_demo(Stokes, a,b,c,d):
    Stokes_cal = np.ones(Stokes.shape)
    q_meas = Stokes[...,1]
    u_meas = Stokes[...,2]
    H = q_meas.shape[0]
    V = q_meas.shape[1]
    a1, a2, a3 = cv2.split(np.moveaxis(a,0,-1))
    b1, b2, b3 = cv2.split(np.moveaxis(b,0,-1))
    c1, c2, c3 = cv2.split(np.moveaxis(c,0,-1))
    d1, d2, d3 = cv2.split(np.moveaxis(d,0,-1))

    ab1 = a1-b1
    cd1 = c1-d1
    ab2 = a2-b2
    cd2 = c2-d2
    ab3 = a3-b3
    cd3 = c3-d3
    for i in range(H):
        for j in range(V):
            M = np.array([[ab2[i,j], ab3[i,j]],
                          [cd2[i,j], cd3[i,j]]])
            #Min = sc.linalg.pinv2(M)
            Min = inv(M)
            S =  np.array([  [ q_meas[i,j]-ab1[i,j] ],
                             [ u_meas[i,j]-cd1[i,j] ]  ])
            Scal = np.matmul(Min, S)
            Stokes_cal[i,j,1]=Scal[0]
            Stokes_cal[i,j,2]=Scal[1]
    return  Stokes_cal

def norm_stokes(Stokes):
    Stokes_norm = copy.deepcopy(Stokes)
    Intensity = copy.deepcopy(Stokes[...,0])
    Stokes_norm[...,0] = Stokes_norm[...,0] / Intensity
    Stokes_norm[..., 1] = Stokes_norm[..., 1] / Intensity
    Stokes_norm[..., 2] = Stokes_norm[..., 2] / Intensity
    return Stokes_norm

def load_xmat(cam_id):
    PolCal_dir = f"C:/Users/masadatz/Google Drive/CloudCT/svs_vistek/calibration/calibration params/polcal_params/"
    X_mat = np.load(PolCal_dir + 'polcal_matrix_' + cam_id + '.npy')
    return X_mat

def main2():

    #for simulation
    #simulated images for finding calibration parameter matrices

    AoLP_deg  =np.array(range(-90, 90, 10))
    AoLP_true = np.mod(np.deg2rad(AoLP_deg),np.pi)
    GT_Snorm = sim_GT_Snorm(AoLP_true, haxis = 100, vaxis = 100)
    error_std = 3
    error_by_type = np.random.normal(0.0, error_std, 4)
    noisy = sim_Noisy_images(GT_Snorm, error_by_type, av_n = 60)
    #I = 0.25 * (noisy[:, 1::2, 1::2] + noisy[:, 0::2, 1::2] + noisy[:, 0::2, 0::2] + noisy[:, 1::2, 0::2])
    #If = np.repeat(np.repeat(I, repeats=2, axis=1), repeats=2, axis=2)
    #noisy = noisy/ If


    #find calibration matrix
    #a, b, c, d = Cal_params_after_demo(noisy, GT_Snorm)
    X_mat = Cal_params(noisy, GT_Snorm)
    AoLP_deg = np.array(range(-65, 65, 10))
    AoLP_val = np.mod(np.deg2rad(AoLP_deg), np.pi)
    DoLP_val = np.array([0.05,0.1,0.2, 0.5, 0.7, 0.9])

    m_error_0 = np.zeros([AoLP_val.shape[0], DoLP_val.shape[0], 2])
    m_error_1 = np.zeros([AoLP_val.shape[0], DoLP_val.shape[0], 2])

    stat_n = 1
    for stat in range(stat_n):
    #simulate images for validation

        error_0 = np.zeros([AoLP_val.shape[0],DoLP_val.shape[0],2])
        error_1 = np.zeros([AoLP_val.shape[0],DoLP_val.shape[0],2])

        for AoLP_n in  range(AoLP_val.shape[0]):
            for DoLP_n in range(DoLP_val.shape[0]):
                AoLP = AoLP_val[AoLP_n]
                DoLP = DoLP_val[DoLP_n]
                GT_Snorm = sim_GT_Snorm(np.array([AoLP]), DoLP_true=DoLP, haxis=100, vaxis=100)
                noisy = sim_Noisy_images(GT_Snorm, error_by_type, av_n = 3)
                error_0 [AoLP_n,DoLP_n,:]=  np.squeeze(np.array(mean_error(np.array([AoLP]), noisy,GT_Snorm, X_mat,
                                                                             DoLP_true=DoLP)))
                error_1 [AoLP_n,DoLP_n,:]= np.squeeze(np.array(mean_error(np.array([AoLP]), noisy,GT_Snorm, X_mat,
                                                                            DoLP_true =DoLP, calibrate=True)))
        m_error_0 = m_error_0 + error_0
        m_error_1 = m_error_1 + error_1
    #print("initial error", m_error_0)
    #print("Calibrated error", m_error_1)

    plt.figure(1)
    a= []
    for i in range(DoLP_val.shape[0]):
       line=plt.plot(AoLP_deg, np.rad2deg(m_error_0[:,i,0])/stat_n)
       a.append(['DoLP='+str(DoLP_val[i])])
    plt.legend(a)
    plt.show()
    plt.figure(2)
    a = []
    for i in range(DoLP_val.shape[0]):
        plt.plot(AoLP_deg, np.rad2deg(m_error_1[:,i,0])/stat_n)
        a.append(['DoLP=' + str(DoLP_val[i])])
    plt.legend(a)
    plt.show()
    a = []
    plt.figure(3)
    for i in range(AoLP_val.shape[0]):
       plt.plot(DoLP_val, m_error_0[i,:,1]/stat_n)
       a.append(['AoLP=' + str(AoLP_deg[i])])
    plt.legend(a)
    plt.show()
    a = []
    plt.figure(4)
    for i in range(AoLP_val.shape[0]):
        plt.plot(DoLP_val, m_error_1[i,:,1]/stat_n)
        a.append(['AoLP=' + str(AoLP_deg[i])])
    plt.legend(a)
    plt.show()
    a =[]



def main():

    #Images for finding calibration parameter matrices

    AoLP_deg = np.array([20, 30, 40, 60, 70, 80, 90, 110, 140, 170]) #np.arange(0,180,20)
    val = 0
    v = 2448
    h = 2048
    dir1 = r'C:\Users\masadatz\Google Drive\CloudCT\svs_vistek\calibration\\'

    raw = np.zeros([AoLP_deg.size, h , v])
    AoLP_true = np.mod(np.deg2rad(AoLP_deg), np.pi)

    GT_Snorm = sim_GT_Snorm(AoLP_true,haxis = h, vaxis = v)

    ID = ['101933', '101934', '101935', '101936','192900073']
    cam = 1
    cam_id = ID[cam]
    dir = dir1+cam_id+r'\full_scan\fixed\fixed_polcal_'

    i = 0
    for AoP in AoLP_deg:
        pattern = dir+str(np.abs(AoP))+'_' + cam_id + '.npy'
        image = np.load(pattern)
        #crop_patch = crop_patch/np.max(crop_patch)
        images_demosaiced = pa.demosaicing(image)
        img_0, img_45, img_90, img_135 = cv2.split(images_demosaiced)
        Stokes = pa.calcLinearStokes(np.moveaxis(np.array([img_0, img_45, img_90, img_135]), 0, -1),
                                     np.deg2rad([0, 45, 90, 135]))
        Intensity = copy.deepcopy(Stokes[...,0])
        raw[i] = image/Intensity
        i = i + 1

    #find calibration matrix
    #a, b, c, d = Cal_params_after_demo(raw, GT_Snorm)
    X_mat = Cal_params(raw, GT_Snorm)

    image = np.load(dir+str(val)+'_' + cam_id + '.npy')

    images_demosaiced = pa.demosaicing(image)
    img_0, img_45, img_90, img_135 = cv2.split(images_demosaiced)
    # for real images we need normalization

    Stokes = pa.calcLinearStokes(np.moveaxis(np.array([img_0, img_45, img_90, img_135]), 0, -1),
                                 np.deg2rad([0, 45, 90, 135]))
    Stokes_norm = norm_stokes(Stokes)

    Stokes_cal = Cal(Stokes_norm, X_mat)
    #Stokes_cal = Cal_after_demo(Stokes, a, b, c, d)

    DoLP_without = pa.cvtStokesToDoLP(Stokes_norm)
    AoLP_without = pa.cvtStokesToAoLP(Stokes_norm)
    DoLP_with  = pa.cvtStokesToDoLP(Stokes_cal)
    AoLP_with = pa.cvtStokesToAoLP(Stokes_cal)


    vmin = 0
    vmax = 0.2
    plt.imshow(abs(1-DoLP_without), cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax)
    plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=plt.get_cmap('gray')))
    plt.clim(vmin, vmax)
    print(np.mean(DoLP_without))
    print(np.mean(np.rad2deg(AoLP_without)))
    plt.show()
    plt.imshow(abs(1-DoLP_with), cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax)
    plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=plt.get_cmap('gray')))
    plt.clim(vmin, vmax)
    plt.show()
    print(np.mean(DoLP_with))
    print(np.mean(np.rad2deg(AoLP_with)))


    plt.imshow(abs(np.rad2deg(AoLP_without)-val))
    plt.colorbar()
    #plt.clim(vmin, vmax)
    plt.show()
    plt.imshow(abs(np.rad2deg(AoLP_with)-val))
    plt.colorbar()
    #plt.clim(vmin, vmax)
    plt.show()
    np.save(dir1 + '\polcal_matrix_' + cam_id,X_mat)

def main3():


    dir1 = f"C:/Users/masadatz/Google Drive/CloudCT/svs_vistek/calibration/"

    ID = ['101933', '101934', '101935', '101936', '192900073']
    cam = 1
    cam_id = ID[cam]
    dir = dir1 + cam_id + f"/full_scan/fixed/fixed_polcal_"
    dir2 = f"/calibration params/polcal_params/"

    val = 80
    image = np.load(dir+str(val)+'_' + cam_id + '.npy')
    #image_dir =r'C:\Users\masadatz\Google Drive\CloudCT\svs_vistek\Data_From_Experiment\A\A32_45000\8\25_15_37_34_509735_101934.npy'
    #image = np.load(image_dir)
    fixed_image = RadCal.correct_rad(image, cam_id)
    images_demosaiced = pa.demosaicing(fixed_image)
    img_0, img_45, img_90, img_135 = cv2.split(images_demosaiced)
    X_mat = np.load(dir1 + dir2+ 'polcal_matrix_' + cam_id+ '.npy')

    Stokes = pa.calcLinearStokes(np.moveaxis(np.array([img_0, img_45, img_90, img_135]), 0, -1),
                                 np.deg2rad([0, 45, 90, 135]))
    Stokes_norm = norm_stokes(Stokes)

    Stokes_cal = Cal(Stokes_norm, X_mat)
    # Stokes_cal = Cal_after_demo(Stokes, a, b, c, d)

    DoLP_without = pa.cvtStokesToDoLP(Stokes_norm)
    AoLP_without = pa.cvtStokesToAoLP(Stokes_norm)
    DoLP_with = pa.cvtStokesToDoLP(Stokes_cal)
    AoLP_with = pa.cvtStokesToAoLP(Stokes_cal)

    vmin = 0
    vmax = 1
    plt.imshow(abs(DoLP_without), cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax)
    plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=plt.get_cmap('gray')))
    plt.clim(vmin, vmax)
    print(np.mean(DoLP_without))
    print(np.mean(np.rad2deg(AoLP_without)))
    plt.show()
    plt.imshow(abs(DoLP_with), cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax)
    plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=plt.get_cmap('gray')))
    plt.clim(vmin, vmax)
    plt.show()
    print(np.mean(DoLP_with))
    print(np.mean(np.rad2deg(AoLP_with)))


    vmin2= 0
    vmax2 =5
    plt.imshow(abs(np.rad2deg(AoLP_without)-val), cmap=plt.get_cmap('gray'), vmin=vmin2, vmax=vmax2)
    plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=plt.get_cmap('gray')))
    plt.clim(vmin2, vmax2)

    plt.show()
    plt.imshow(abs(np.rad2deg(AoLP_with)-val), cmap=plt.get_cmap('gray'), vmin=vmin2, vmax=vmax2)
    plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=plt.get_cmap('gray')))
    plt.clim(vmin2, vmax2)
    plt.show()
if __name__ == "__main__":
    main3()


