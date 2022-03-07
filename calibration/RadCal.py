import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import json

def spread_by_pol(cover):
    # decompose the scans to the 4 kinds of pixels (0,90,180,270)
    covers = []
    covers.append(cover[::2,::2])
    covers.append(cover[::2,1::2])
    covers.append(cover[1::2,::2])
    covers.append(cover[1::2,1::2])
    return covers

def rad_cal_mat(cover,cam_id,exposure,DN):
    ## Compute the matrice for the flat-field correction such that
    # if we multiply it with the cover elementwise we get all the values to be the mean value of the original cover.
    dir1 = f"C:/Users/masadatz/Google Drive/CloudCT/svs_vistek/calibration/calibration params/radcal_params/"
    mean = np.mean(cover)
    ff_fix = np.full(cover.shape, mean)/cover
    data = {"name": f"{cam_id}",
            "exposure time": exposure,
            "integrating sphere DN": DN,
            "fixer image": ff_fix.tolist() }
    # "C" : C}
    with open(dir1+"{cam_id}_calibration_params.json", "w") as file:
        json.dump(data, file)
    return ff_fix

def fit_curve_by_pix_type(covers):

    #fit a curve for each pixel type

    models = []
    for cov in covers:
        x = []
        y = []
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                if cov[i,j]>100:
                    x.append((i,j))
                    y.append(cov[i,j])
        x = np.array(x)
        y = np.array(y)
        polynomial_features= PolynomialFeatures(degree=6)
        x_poly = polynomial_features.fit_transform(x)
        model = linear_model.LinearRegression()
        model.fit(x_poly, y)
        models.append(model)
        y_poly_pred = model.predict(x_poly)
        rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
        r2 = r2_score(y,y_poly_pred)
        print(rmse)
    return models, polynomial_features

def plot_curves(covers,models, polynomial_features):
    # %%plot the x and y axis with the "predictions" of our curve to estimate the fit.

    for cov, model in zip(covers, models):
        x_poly = polynomial_features.fit_transform(
            np.column_stack([np.full(len(cov[1024 // 2]), 512), np.arange(1224)]))
        print(x_poly.shape)
        print(np.mean(cov[cov > 100]))
        plt.scatter(range(2448 // 2), cov[1024 // 2, :])
        plt.plot(range(2448 // 2), model.predict(x_poly), color='red')
        plt.show()
        x_poly = polynomial_features.fit_transform(
            np.column_stack([np.arange(1024), np.full(len(cov[:, 1224 // 2]), 512)]))

        plt.scatter(range(2048 // 2), cov[:, 1224 // 2])
        plt.plot(range(2048 // 2), model.predict(x_poly), color='red')
        plt.show()

def fill_in_by_model(cover,covers,models):
    # %% fix by predictions

    for cov, model in zip(covers, models):
        x = []
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                if cov[i, j] < 500:
                    x.append((i, j))
        if len(x) > 0:
            x = np.array(x)
            polynomial_features = PolynomialFeatures(degree=6)
            x_poly = polynomial_features.fit_transform(x)
            predictions = model.predict(x_poly)
            for (i, j), prediction in zip(x, predictions):
                cov[i, j] = prediction
        plt.imshow(cov)
        plt.show()
    cover2 = np.zeros(cover.shape)
    cover2[::2, ::2] = covers[0]
    cover2[::2, 1::2] = covers[1]
    cover2[1::2, ::2] = covers[2]
    cover2[1::2, 1::2] = covers[3]
    return cover2

def ff_correct(image, cam_id, exposure):
    dir_ff = f"C:/Users/masadatz/Google Drive/CloudCT/svs_vistek/calibration/calibration params/radcal_params/"
    dir_dark = f"C:/Users/masadatz/Google Drive/CloudCT/svs_vistek/calibration/calibration params/DarkNoise/"
    dark_cover = np.load(dir_dark+f"camera_{cam_id}\DarkNoise_CAM{cam_id}_{exposure}ET.npy")

    with open(dir_ff+f"{cam_id}_calibration_params.json", "rb") as file:
        param_dict = json.load(file)
        cover_fix = np.array(param_dict["fixer image"])# needs to be a function by exposure time
        fixed_image = np.clip(image-dark_cover,0,4096) * cover_fix #
    return fixed_image

def gray2rad(image,cam_id,exposure,bit):
    dir_ff = f"C:/Users/masadatz/Google Drive/CloudCT/svs_vistek/calibration/calibration params/radcal_params/"

    with open(dir_ff + f"{cam_id}_calibration_params.json", "rb") as file:
        param_dict = json.load(file)
        C = np.array(param_dict["C"])
        exposure_origin = param_dict["exposure time"]
    if (bit==8):
        C = C/256
    rad = exposure_origin*image/(exposure*C) #  [W/m^2/sr]      *10**(6)??
    return rad

def main():
    ID = ['101933', '101934', '101935', '101936', '192900073']
    cam = 1
    cam_id = ID[cam]
    exposure = 3000
    dir = r'C:/Users/masadatz/Google Drive/CloudCT/svs_vistek/calibration/101934/full_scan/polcal_40_101934.npy'  #for example
    image0 = np.load(dir)
    image1 = ff_correct(image0,cam_id,exposure)
    plt.imshow(image0, cmap=plt.get_cmap('gray'), vmin=0, vmax=4096)
    plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=plt.get_cmap('gray')))
    plt.show()
    plt.imshow(image1, cmap=plt.get_cmap('gray'), vmin=0, vmax=4096)
    plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=plt.get_cmap('gray')))
    plt.show()

if __name__ == '__main__':
    main()