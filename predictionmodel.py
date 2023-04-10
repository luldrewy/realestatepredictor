''' Prototype 1 (predicting with sample data)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from scrapehouseprice import Scraper


file = "/Users/andrew/Desktop/LearningC++/LearningPython/MachineLearning/test.csv"
df1 = pd.read_csv(file)

    
#fill all the NaN values in the lot_size column with the median 
df1[["lot_size"]] = df1[["lot_size"]].fillna(df1[["lot_size"]].median())


#make a list of which index has an a nan value in the lot_size_units
nanValues = [i for i in range(len(df1["lot_size_units"])) if "nan" in str(df1["lot_size_units"][i])]



#this loop transforms all the nans into either acres or sqft
for i in nanValues:  #for every index value of the list of Nans, check if the ith row in the lot_size column is under 250 acres
                    # if the ith row in the lot_size_column is a value under 250 it is in acres
    if df1["lot_size"][i] < 250: #250 is the highest number of acres and 500 is the lowest number of sqr foot
        df1["lot_size_units"][i] = "acre"
    else:
        df1["lot_size_units"][i] = "sqft"



#gets the list of all indexes that have acres in the lot_size_units
containsAcres = [i for i in range(len(df1["lot_size_units"])) if "acre" in str(df1["lot_size_units"][i])]
#gets the list of values in the ith row in the lot_size column
acres = [df1.loc[i]["lot_size"] for i in containsAcres]

#convert the acres into sqrft
for i in containsAcres:
    df1.loc[i,"lot_size"] = df1.loc[i,"lot_size"] * 43560
    df1.loc[i,"lot_size_units"] = "sqft"

#print(df1["lot_size_units"].str.contains("acre").any()) check to see if there are any acres left


#get label encoding for zip codes: I am using label encoding instead of one hot encoding because 
# I believe the zip codes are hierachal, meaning that there is a ranking between the zip codes (certain zip codes are morre expensive)

le = LabelEncoder()

df1["Zip Code"] = le.fit_transform(df1["zip_code"])

df1["price per sqft"] = df1["price"] / df1["size"]


#print(df1[df1["size"]/ df1["beds"]< 300].head())
df1 = df1[~(df1["size"]/ df1["beds"]< 300)]


df1 = df1[~(df1["price per sqft"] < 200)] #assume that houses with price per square foot under 200 dollars are outliers, so get rid of them


def removeOutliers(dataFrame):
    dfOutlier = pd.DataFrame()
    for key, subDf in dataFrame.groupby("Zip Code"):
        m = np.mean(subDf["price per sqft"])
        st = np.std(subDf["price per sqft"])
        reducedDf = subDf[(subDf["price per sqft"] > (m-st)) & (subDf["price per sqft"] <= (m+st))] #reducedDf gets all the data that are above one standard deviation to the left of 
                                                                                                    #the mean and less than one standard deviation to the right of the mean
        dfOutlier = pd.concat([dfOutlier, reducedDf], ignore_index= True)
    
    return dfOutlier

df1 = removeOutliers(df1)
#print(df1)


def plotScatterChart(df, zipCode):
    #print(df[(df["zip_code"] == zipCode) & (df["beds"] == 2)])
    twoBeds = df[(df["zip_code"] == zipCode) & (df["beds"] == 2)]
    threeBeds = df[(df["zip_code"] == zipCode) & (df["beds"] == 3)]

    plt.scatter(twoBeds["size"], twoBeds["price"],color = "green",s=50)
    plt.scatter(threeBeds["size"], threeBeds["price"], color = "red",s =50)
    plt.xlabel("total square foot area")
    plt.ylabel("price")
    plt.title(zipCode)
    plt.show()

#plotScatterChart(df1, 98122)

def removeBedOutliers(dataframe):
    outlier = np.array([])
    for zipCode, zipCodeDf in dataframe.groupby("zip_code"): #for each zip_code value and the new (sorted by zip_code) dataframes, each containing rows with same zipcode, called zipCodeDf...
        bedStats = {}
        for bhk, bhkDf in zipCodeDf.groupby("beds"): #for each bed value (ie. 1, 2, or 3 beds) and the individual dataframes sorted by beds of the entire dataframe...
            bedStats[bhk] = {   #create the key in the dictionary as the bed value
                "mean" : np.mean(bhkDf["price per sqft"]), # within that key, the value will store another set of keys (nested dictionaries)
                                                            # the first key will be the mean which is then paired with the value, the average of the "price per sqft" column in the bhk Dataframe
                "std" : np.std(bhkDf["price per sqft"]), #the second key will be the standard deviation which is paired with the value, the standard deviation of the "price per sqft column" in the bhk Dataframe
                "count" : bhkDf.shape[0]
            }

        for bhk, bhkDf in zipCodeDf.groupby("beds"):
            stats = bedStats.get(bhk - 1) #why do we substract by 1
            if stats and stats["count"] > 5: #what is the count meaning
                outlier = np.append(outlier, bhkDf[bhkDf["price per sqft"] < (stats["mean"])].index.values)

    return dataframe.drop(outlier, axis = "index")


df1 = removeBedOutliers(df1)
df1 = df1[df1["beds"] + 2 > df1["baths"]] #drop the houses with more baths than the number of beds plus 2 because those houses are abnormal 



x = df1[["beds", "baths", "size", "lot_size", "Zip Code"]] #set the input variables to be all the columns with relevant features

y = df1[["price"]] #set the output variables to be the price column which is the target variable

xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size= 0.8)


def findBestModel(x,y):
    methods = {
        "linear regression": {
            "model" : LinearRegression(),
            "params" : {
                "fit_intercept" : [True, False]
            }
        },
        "decision tree" : {
            "model" : DecisionTreeRegressor(),
            "params" : {
                "criterion" : ["squared_error", "friedman_mse"],
                "splitter" : ["best","random"]
            }
        },
        "lasso" : {
            "model" : Lasso(),
            "params" : {
                "alpha" : [1,2],
                "selection" : ["cyclic", "random"]
            }
        }
        
    }
    myList = []
    for algo, algoParams in methods.items():
        myModel = GridSearchCV(algoParams["model"], param_grid= algoParams["params"], cv= 5, return_train_score=False)
        myModel.fit(x,y)

        myList.append([algo, 
                       myModel.best_score_,
                       myModel.best_params_])
        
    return pd.DataFrame(myList, columns= ["model", "best score", "best params"])

print(findBestModel(x,y))


#indexNumber = df1[df1["zip_code"] == 98102].index.values
#series = np.array(df1.loc[indexNumber]["Zip Code"]) 
#print(series[0])

print(df1.head())

myModel = LinearRegression(fit_intercept= True) 
myModel.fit(xTrain, yTrain)

def predictPrice(beds, baths, size, lot_size, zipCode):
    indexNumber = df1[df1["zip_code"] == zipCode].index.values
    series = np.array(df1.loc[indexNumber]["Zip Code"]) 
    #print(series[0])

    return myModel.predict([[beds, baths, size, lot_size, series[0]]])
    

#print(predictPrice(2,3,970,593,98102))
print(myModel.score(xTest, yTest))

'''
#Prototype 2 (Prototyping with Data scraped from Zillow)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from scrapehouseprice import Scraper


def main(df, userInputbeds, userInputbaths, userInputsize, userInputZipCode):
    #file = "/Users/andrew/Desktop/LearningC++/LearningPython/MachineLearning/ZillowData.csv"
    df1 = df #pd.read_csv(file)

    #fill all the NaN values in the lot_size column with the median 
    #df1[["lot_size"]] = df1[["lot_size"]].fillna(df1[["lot_size"]].median())
    #remove all the rows that contain NaN values 
    for feature in ["beds","baths","size","zip_code","price"]: 
        df1 = df1[~df1[feature].isna()]




    #make a list of which index has an a nan value in the lot_size_units
    #nanValues = [i for i in range(len(df1["lot_size_units"])) if "nan" in str(df1["lot_size_units"][i])]



    #this loop transforms all the nans into either acres or sqft
    #for i in nanValues:  #for every index value of the list of Nans, check if the ith row in the lot_size column is under 250 acres
                        # if the ith row in the lot_size_column is a value under 250 it is in acres
    #    if df1["lot_size"][i] < 250: #250 is the highest number of acres and 500 is the lowest number of sqr foot
    #        df1["lot_size_units"][i] = "acre"
    #    else:
    #        df1["lot_size_units"][i] = "sqft"



    #gets the list of all indexes that have acres in the lot_size_units
    #containsAcres = [i for i in range(len(df1["lot_size_units"])) if "acre" in str(df1["lot_size_units"][i])]
    #gets the list of values in the ith row in the lot_size column
    #acres = [df1.loc[i]["lot_size"] for i in containsAcres]

    #convert the acres into sqrft
    #for i in containsAcres:
    #    df1.loc[i,"lot_size"] = df1.loc[i,"lot_size"] * 43560
    #    df1.loc[i,"lot_size_units"] = "sqft"

    #print(df1["lot_size_units"].str.contains("acre").any()) check to see if there are any acres left


    #get label encoding for zip codes: I am using label encoding instead of one hot encoding because 
    # I believe the zip codes are hierachal, meaning that there is a ranking between the zip codes (certain zip codes are morre expensive)

    le = LabelEncoder()

    df1["Zip Code"] = le.fit_transform(df1["zip_code"])

    df1["price per sqft"] = df1["price"] / df1["size"]


    #print(df1[df1["size"]/ df1["beds"]< 300].head())
    df1 = df1[~(df1["size"]/ df1["beds"]< 300)]


    df1 = df1[~(df1["price per sqft"] < 200)] #assume that houses with price per square foot under 200 dollars are outliers, so get rid of them


    def removeOutliers(dataFrame):
        dfOutlier = pd.DataFrame()
        for key, subDf in dataFrame.groupby("Zip Code"):
            m = np.mean(subDf["price per sqft"])
            st = np.std(subDf["price per sqft"])
            reducedDf = subDf[(subDf["price per sqft"] > (m-st)) & (subDf["price per sqft"] <= (m+st))] #reducedDf gets all the data that are above one standard deviation to the left of 
                                                                                                        #the mean and less than one standard deviation to the right of the mean
            dfOutlier = pd.concat([dfOutlier, reducedDf], ignore_index= True)
        
        return dfOutlier

    df1 = removeOutliers(df1)
    #print(df1)


    def plotScatterChart(df, zipCode):
        #print(df[(df["zip_code"] == zipCode) & (df["beds"] == 2)])
        twoBeds = df[(df["zip_code"] == zipCode) & (df["beds"] == 2)]
        threeBeds = df[(df["zip_code"] == zipCode) & (df["beds"] == 3)]

        plt.scatter(twoBeds["size"], twoBeds["price"],color = "green",s=50)
        plt.scatter(threeBeds["size"], threeBeds["price"], color = "red",s =50)
        plt.xlabel("total square foot area")
        plt.ylabel("price")
        plt.title(zipCode)
        plt.show()

    #plotScatterChart(df1, 98122)

    def removeBedOutliers(dataframe):
        outlier = np.array([])
        for zipCode, zipCodeDf in dataframe.groupby("zip_code"): #for each zip_code value and the new (sorted by zip_code) dataframes, each containing rows with same zipcode, called zipCodeDf...
            bedStats = {}
            for bhk, bhkDf in zipCodeDf.groupby("beds"): #for each bed value (ie. 1, 2, or 3 beds) and the individual dataframes sorted by beds of the entire dataframe...
                bedStats[bhk] = {   #create the key in the dictionary as the bed value
                    "mean" : np.mean(bhkDf["price per sqft"]), # within that key, the value will store another set of keys (nested dictionaries)
                                                                # the first key will be the mean which is then paired with the value, the average of the "price per sqft" column in the bhk Dataframe
                    "std" : np.std(bhkDf["price per sqft"]), #the second key will be the standard deviation which is paired with the value, the standard deviation of the "price per sqft column" in the bhk Dataframe
                    "count" : bhkDf.shape[0]
                }

            for bhk, bhkDf in zipCodeDf.groupby("beds"):
                stats = bedStats.get(bhk - 1) #why do we substract by 1
                if stats and stats["count"] > 5: #what is the count meaning
                    outlier = np.append(outlier, bhkDf[bhkDf["price per sqft"] < (stats["mean"])].index.values)

        return dataframe.drop(outlier, axis = "index")






    x = df1[["beds", "baths", "size", "Zip Code"]] #set the input variables to be all the columns with relevant features

    y = df1[["price"]] #set the output variables to be the price column which is the target variable

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size= 0.8)


    def findBestModel(x,y):
        methods = {
            "linear regression": {
                "model" : LinearRegression(),
                "params" : {
                    "fit_intercept" : [True, False]
                }
            },
            "decision tree" : {
                "model" : DecisionTreeRegressor(),
                "params" : {
                    "criterion" : ["squared_error", "friedman_mse"],
                    "splitter" : ["best","random"]
                }
            },
            "lasso" : {
                "model" : Lasso(),
                "params" : {
                    "alpha" : [1,2],
                    "selection" : ["cyclic", "random"]
                }
            },
            "ridge" : {
                "model" : Ridge(),
                "params" : {
                    "alpha" : [1,2],
                    "solver" : ["auto", "svd"]
                }
            }
            
        }
        myList = []
        for algo, algoParams in methods.items():
            myModel = GridSearchCV(algoParams["model"], param_grid= algoParams["params"], cv= 5, return_train_score=False)
            myModel.fit(x,y)

            myList.append([algo, 
                        myModel.best_score_,
                        myModel.best_params_])
            
        return pd.DataFrame(myList, columns= ["model", "best score", "best params"])

    print(findBestModel(x,y))


    #indexNumber = df1[df1["zip_code"] == 98102].index.values
    #series = np.array(df1.loc[indexNumber]["Zip Code"]) 
    #print(series[0])

    #print(df1.head())

    myModel = LinearRegression(fit_intercept= True) 
    myModel.fit(xTrain, yTrain)

    def predictPrice(beds, baths, size, zipCode):
        indexNumber = df1[df1["zip_code"] == zipCode].index.values
        series = np.array(df1.loc[indexNumber]["Zip Code"]) 
        #print(series[0])
        print(myModel.score(xTest, yTest))
        return myModel.predict([[beds, baths, size, series[0]]])
    
    return predictPrice(userInputbeds, userInputbaths, userInputsize, userInputZipCode)

    #print(predictPrice(3,3,1700,90012))



