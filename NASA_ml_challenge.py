import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import random


def fun(x, a, b):
    # return a * (np.exp(-b * x) - 1)
    return a * (1- np.exp(-b * x))


def norm(x):
    M= max(x)
    m= min(x)
    r= M-m

    return [(i-m)/r for i in x]

def main( ):

    # Training phase
    file= "NASAEngineData.xlsx"
    xls=  pd.ExcelFile(file)
    df= pd.read_excel(xls, 'Historical Data')

    unit= df['unit number'].values
    unit = unit[~np.isnan(unit)]
    L= len(unit)
    RUL= df['RUL'].values
    RUL= RUL[~np.isnan(RUL)]

    # op1= df['operational setting 1'].values
    # op1 = op1[~np.isnan(op1)]
    #
    # op2= df['operational setting 1'].values
    # op2 = op2[~np.isnan(op2)]

    # op3= df['operational setting 1'].values
    # op3 = op3[~np.isnan(op3)]

    # fig= plt.figure( )
    # ax= fig.add_subplot(111, projection='3d')
    # ax.scatter(op1,op2,op3)
    # plt.figure( )
    # plt.plot(op1,op2)
    # plt.xlabel("Operational setting 1")
    # plt.ylabel("Operational setting 2")
    # plt.show(block= False)

    for i in range(1,16):
        plt.figure()
        temp= df['Sensor Measurement '+str(i)].values
        temp= temp[~np.isnan(temp)]
        plt.scatter(RUL,temp)
        plt.title('Sensor Measurement '+str(i))
        plt.show(block= False)
    # Indentified the sensors from above plots, that show a pattern through engine's life cycle

    # Read the important sensor columns
    S= np.ndarray(shape=(L,0))
    # sen= [2,3,4,7,11,12,15,20,21]
    # sen = [2, 3, 4, 7, 11, 12, 15]
    sen= [7, 12, 15]

    num_sen= len(sen)
    for i in sen:
        temp= df['Sensor Measurement ' + str(i)].values
        temp = temp[~np.isnan(temp)].reshape(L, 1)
        temp= norm(temp)
        S= np.hstack((S,temp))


    # Linear regression model fitting for obtaining health index (HI)
    X = np.ndarray(shape=(0, num_sen))
    Ti= [ ]
    y= np.ndarray(shape=(0,1))
    M= np.ndarray(shape=(100,2))
    for i in range(1,101):
        ind= np.where(unit==i)[0]
        Ti.append(len(ind)) # To be used later in the testing phase

        temp= np.linspace(1, 0, len(ind), dtype=float).reshape(len(ind),1)
        y= np.vstack((y,temp))

        X= np.vstack((X,S[ind,: ]))


        # Exponential curve fitting
        C_adj = np.arange(-len(ind),0,1)+1
        popt, _ = curve_fit(fun, C_adj, temp.flatten())

        M[i-1, 0] = popt[0]
        M[i-1, 1] = popt[1]


    lm= LinearRegression()
    lm.fit(X,y)
    # lm.coef_ and lm.intercept_ to be used later in testing


    # Validation phase
    error= 0
    val_set= np.random.randint(1,101,15) # Validation set has 15 engines

    for i in range(len(val_set)):

        r= int(random.uniform(0.5,0.7)*Ti[val_set[i]-1]) # We are letting the algorithm figure out the rest life cycles
        ind = np.where(unit == val_set[i])[0]
        SV= S[ind[ :r],: ]

        prediction= RUL_finding(sen, SV, lm, M, Ti, r)
        error+= error_estimation(prediction,Ti[val_set[i]-1]-r+1)

    print("Validation mean absolute error: ", error/len(val_set))
    print("\n")


    # Testing phase
    test_set= [1, 2, 3, 4, 24, 25, 34]
    given= [112, 98, 69, 82, 20, 145, 45]
    engine_name= ['Engine '+str(i) for i in test_set]

    error= 0
    for i in range(len(test_set)):
        prediction= eval_test(sen, Ti, xls, engine_name[i], M, lm)
        error+= error_estimation(prediction,given[i])

    print("\n")
    print("Test mean absolute error: ", error/len(test_set))

# --------------------------------------------------------------------------------------------

def eval_test(sen, Ti, xls, test_engine, M, lm):

    df = pd.read_excel(xls, test_engine)
    unit= df["unit number"].values
    unit = unit[~np.isnan(unit)]
    r= len(unit)

    S = np.ndarray(shape=(r, 0))
    for i in sen:
        temp= df['Sensor Measurement ' + str(i)].values
        temp = temp[~np.isnan(temp)].reshape(r, 1)
        temp= norm(temp)
        S= np.hstack((S,temp))


    RUL_final= RUL_finding(sen, S, lm, M, Ti, r)
    print("Given", r, "cycles of", test_engine, ", the RUL is", RUL_final)
    return RUL_final


def RUL_finding(sen, S,lm, M, Ti,r):

    y= S @ lm.coef_.reshape(len(sen),1) + lm.intercept_
    RUL_pred= np.zeros((1,100),dtype=float)[0]
    D= np.zeros((1,100),dtype=float)[0]

    for i in range(100):
        d= 0
        for j in range(r):

            d += (y[j] - fun(-Ti[i] + j, M[i, 0], M[i, 1])) ** 2

        RUL_pred[i]= Ti[i]-r+1
        D[i]= d

    # D_pos = D[np.where(RUL_pred>0)]
    # RUL_pos_pred= RUL_pred[np.where(RUL_pred>0)]


    # Outlier removal
    D_pos = [ ]
    RUL_pos_pred= [ ]

    for i in range(len(RUL_pred)):

        if RUL_pred[i]>0 and RUL_pred[i]+r>150 and RUL_pred[i]<190:
            D_pos.append(D[i])
            RUL_pos_pred.append(RUL_pred[i])


    temp= sorted(zip(D_pos, RUL_pos_pred))
    D_pos, RUL_pos_pred = map(list, zip(*temp))

    # Soft voting

    # RUL_final= 0
    # K= 10 # number of nearest neighbors
    # for k in range(1,K+1):
    #     RUL_final+= 1/k*RUL_pos_pred[k]
    #
    # RUL_final/= sum(1/k for k in range(1,K+1))


    RUL_final= 4/5*RUL_pos_pred[0]+1/5*RUL_pos_pred[-1]

    return RUL_final


def error_estimation(prediction,given):
    d= prediction- given
    # return np.exp(-d/13)-1 if d<=0 else np.exp(d/10)-1
    return abs(d)


if __name__== "__main__":
    main( )