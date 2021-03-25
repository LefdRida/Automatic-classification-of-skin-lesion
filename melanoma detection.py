# Attention : il faut enregistrer ce fichier dans un dossier qui contient les images et le fichier data.csv
# sinon vous auriez une erreur lors de l'éxecution.

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy import ndimage
import scipy
import cv2
import imageio
from skimage.feature import local_binary_pattern 
from skimage.measure import label, regionprops, regionprops_table
import skimage.measure
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tqdm import tqdm
from IPython.display import SVG
from sklearn.metrics import accuracy_score


# Import the name of the images and their labels 
data = pd.read_csv("data.csv", sep=",")
newdata = data[0:421]
Y = newdata['melanoma']
name = newdata[["image_id"]]
newY = []
ext = ".jpg"

for i in range(421):
    try : 
        I = imageio.imread(name.loc[i].image_id + ext)
        newY.append(Y[i].astype(int))
    except FileNotFoundError:
        pass
Y = pd.DataFrame(newY)   # Y a dataframe contains the label of each image

 
# Functions for features extraction

def LPB(I): 
    B = np.zeros(np.shape(I)) 
    code = np.array([[1,2,4],[8,0,16],[32,64,128]]) 
    
    for i in np.arange(1, I.shape[0] - 2):
        for j in np.arange(1, I.shape[1] - 2):
            w = I [ i - 1: i +2, j - 1: j +2]
            w = w >= I[ i , j ]
            w = w*code
            B[i, j] = np.sum(w)
    h,edges = np.histogram(B[1: - 1, 1: - 1], density=True, bins=256);
    
    return h
#########################################################################################################
def get_tumour(I,B):
    
    I1 = np.multiply(I[:,:,0],np.logical_not(B))
    I2 = np.multiply(I[:,:,1],np.logical_not(B))
    I3 = np.multiply(I[:,:,2],np.logical_not(B))
    
    I[:,:,0] = I[:,:,0] - I1
    I[:,:,1] = I[:,:,1] - I2
    I[:,:,2] = I[:,:,2] - I3
    
    return I
#########################################################################################################
def extract_prop(B):
    
    prop = []
    label_img = label(B) # B : binary segmented image
    props = regionprops_table(label_img, properties=['area','convex_area','bbox_area','eccentricity','extent','filled_area'\
                                    ,'minor_axis_length','major_axis_length','solidity','perimeter','equivalent_diameter'])
    
    prop.append(props['area'][0])
    prop.append(props['perimeter'][0])
    prop.append(props['convex_area'][0])
    prop.append(props['bbox_area'][0]) 
    prop.append(props['eccentricity'][0])
    prop.append(props['extent'][0])
    prop.append(props['filled_area'][0])
    prop.append(props['minor_axis_length'][0])
    prop.append(props['major_axis_length'][0])
    prop.append(props['solidity'][0])
    prop.append(props['equivalent_diameter'][0])
   
    return np.array(prop)

###########################################################################################################

def color_variegation(I):
    
    I = I/255
    m, n, k = I.shape
    white, black, red, light_brown, dark_brown, blue_gray = 0, 0, 0, 0, 0, 0
    
    for i in range(m):
        for j in range(n):
            R, G, B = I[i,j,0],I[i,j,1],I[i,j,2]
            if (R > 0.8 and G > 0.8 and B > 0.8) : 
                white +=1
            if (R >= 0.588 and G < 0.2 and B < 0.2) : 
                red +=1
            if ((R >= 0.588 and R <= 0.94) and (G > 0.2 and G <= 0.588) and (B > 0 and B < 0.392)) : 
                light_brown +=1
            if ((R > 0.243 and R < 0.56) and G < 0.392 and (B > 0 and B < 0.392)) : 
                dark_brown +=1
            if (R <= 0.588 and (G >= 0.392 and G <= 0.588) and (B <= 0.588 and B >= 0.490)) : 
                blue_gray +=1
            if (0 < R <= 0.243 and 0 < G <= 0.243 and 0 < B <= 0.243) : 
                black +=1
    
    number_pixels = m*n
    colors = [white,red,light_brown,dark_brown,blue_gray,black]
    C = 0
    
    for i in range(6):
        if colors[i] > 0.05*number_pixels:
            C += 1
    
    return C

###########################################################################################################

def super_pixels(I):
    
    # the blue component
    b = np.array(I[:,:,2])
    b = np.reshape(r, (1,-1))
    b = r[0]
    
    #the green component
    g = np.array(I[:,:,1])
    g = np.reshape(g, (1,-1))
    g = g[0]
    
    #indexes of the last layer which are significative
    indexes = [i for i in range(len(g)) if g[i] == 3]
    
    #extraction of the superpixels in this layer
    r_bis = r[indexes]
    
    #Removing repeated values
    r_sub = list(dict.fromkeys(r[indexes]))
    
    return max(r_sub)

# extract the LBP histogram of the blue component for each original image

ext = ".jpg"
seg = "_segmentation.png"
sup = "_superpixels.png"

from tqdm import tqdm

name = newdata[["image_id"]]
I = imageio.imread(name.loc[1].image_id + ext)

lpb = []

for i in tqdm(range(421)):
    try : 
        
        I = cv2.resize(imageio.imread(name.loc[i].image_id + ext),(200,200))
        
        texture = LPB(I[:,:,2])
        
        lpb.append(texture)
    
    except FileNotFoundError :
        pass

# extract the geometrical features from the binary images

ext = ".jpg"
seg = "_segmentation.png"
sup = "_superpixels.png"


from tqdm import tqdm

name = newdata[["image_id"]]

geometrical_features = []

for i in tqdm(range(421)):
    try : 
 
        B = cv2.resize(imageio.imread(name.loc[i].image_id + seg),(300,300))

        prop = extract_prop(B)

        geometrical_features.append(prop)
        
    except FileNotFoundError :
        pass
    
# extract the color variegation feature and the number of superpixels 

ext = ".jpg"
seg = "_segmentation.png"
sup = "_superpixels.png"

from tqdm import tqdm

name = newdata[["image_id"]]
other_features = []

for i in tqdm(range(421)):
    try : 
        
        I = cv2.resize(imageio.imread(name.loc[i].image_id + ext),(200,200))
        B = cv2.resize(imageio.imread(name.loc[i].image_id + seg),(200,200))
      
        
        I = get_tumour(I,B)
        T = imageio.imread(name.loc[i].image_id + sup)
        
        C = color_variegation(I)
        red = super_pixels(T)
        
        features = np.concatenate((C, red), axis=None)
        other_features.append(features)
        
    except FileNotFoundError :
        pass
    
# split data into training set and test set 

features = np.concatenate((geometrical_features ,lpb,other_features), axis=1)
df = pd.DataFrame(features) # a data frame contains all the features

X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.25, random_state=42) # splitting the data


# building the classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.tree

#Logestic Regression
LR_m = LogisticRegression().fit(X_train, np.ravel(y_train))
y_pred_LR = LR_m.predict(X_test)

#SVM
SVM_m = make_pipeline(StandardScaler(), SVC(gamma='auto')).fit(X_train, np.ravel(y_train))
y_pred_SVM = SVM_m.predict(X_test)

#Gaussian naive bayes
gnb_m = GaussianNB().fit(X_train, y_train)
y_pred_nb = gnb_m.predict(X_test)

#Decision tree
tree_m = sklearn.tree.DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
y_pred_tree = tree_m.predict(X_test)

# stochastic gradient descent
gradient_descent_m = SGDClassifier().fit(X_train, y_train)
y_pred_gd = gradient_descent_m .predict(X_test)

# compute the accuracy and the Fscore of each classifier

from sklearn.metrics import accuracy_score, f1_score

acc_LR = accuracy_score(y_test,y_pred_LR)
f1_LR = f1_score(y_test,y_pred_LR)

acc_SVM = accuracy_score(y_test,y_pred_SVM)
f1_SVM = f1_score(y_test,y_pred_SVM)

acc_nb = accuracy_score(y_test,y_pred_nb)
f1_nb = f1_score(y_test,y_pred_nb)

acc_tree = accuracy_score(y_test,y_pred_tree)
f1_tree = f1_score(y_test,y_pred_tree)

acc_gd = accuracy_score(y_test,y_pred_gd)
f1_gd = f1_score(y_test,y_pred_gd)

#Plot of accuracy
fig=plt.figure()
plt.bar([1,2,3,4,5],height=[acc_LR,acc_SVM,acc_nb,acc_tree,acc_gd],tick_label=['Logitic \n regression', 'SVM', 'Naive \n Bayesian','Decision \n tree', 'Gradient \n descent'])
plt.title('models accuracy')
plt.ylabel('accuracy')
plt.xlabel('model')
plt.show()
plt.bar([1,2,3,4,5],height=[f1_LR,f1_SVM,f1_nb,f1_tree,f1_gd],tick_label=['Logitic \n regression', 'SVM', 'Naive \n Bayesian','Decision \n tree', 'Gradient \n descent'])
plt.title('models F-score')
plt.ylabel('fscore')
plt.xlabel('model')
plt.show()


    