# Importing all required Libraries
#importing pandas to read and manipulate csv files
import pandas as pd 
#importing numpy to work with arrays faster
import numpy as np 
#importing k means to make clusters for data.
from sklearn.cluster import KMeans  
#importing matplotlib to plot multiple graphs
import matplotlib.pyplot as plots
#importing labelencoder to encode categoical data
from sklearn.preprocessing import LabelEncoder
#importing minmaxscaler to scale data 
from sklearn.preprocessing import MinMaxScaler
#importing cdist to calculate distance between inputs
from scipy.spatial.distance import cdist 
# import scipy.
import scipy
# importing curve fit from scipy.
from scipy.optimize import curve_fit
#importing stats here
from scipy import stats  
# importing warnings.
import warnings 
# filtering unncessary warnings
warnings.filterwarnings('ignore')     

# Loading the rural population dataset
"""In this rural population data of all over world from 1960 ro 2021 is used. In this code, first the code is
pre-processed with preprocessing methods. After preprocessing clusters are made using k means technique.
In this code multiple graphs are shown for population growth in any country. At last a curve fit is shown 
which is the best fit graph."""

# Creates the function to analyse the dataset.
#the rural population data contains unnecessary rows and columns which are removed by this function.
def reading_dataset(new_file):
    #reading the dataset and skipping starting four rows
    rural_population_data = pd.read_csv(new_file, skiprows=4) # using pandas read data and skip starting 4 rows from data.
    #dropping unnecessary columns
    rural_population_data1 = rural_population_data.drop(['Unnamed: 66', 'Indicator Code',  'Country Code'],axis=1) # dropping the columns.
    #setting country name column as index
    rural_population_data2 = rural_population_data.set_index("Country Name")  
    #transposing the dataset
    rural_population_data2=rural_population_data2.T 
    #resetting the index
    rural_population_data2.reset_index(inplace=True)
    #renaming year as index 
    rural_population_data2.rename(columns = {'index':'Year'}, inplace = True) 
    #returning the simple and transposed dataset
    return rural_population_data1, rural_population_data2 

# defining the file path for rural population csv data
new_file = '/content/drive/MyDrive/rural data/API_SP.RUR.TOTL_DS2_en_csv_v2_5363648.csv'  
#creating rural population simple dataframe and transposed dataframe
Rural_population_data, Rural_population_transposed_data = reading_dataset(new_file)   
#showing starting rows of datset
Rural_population_data.head()

Rural_population_transposed_data.head()
Rural_population_data.shape, Rural_population_transposed_data.shape

# Extracting year data and dropping null values with the help of function.
def Rural_population_data2(Rural_population_data): 
    Rural_population_data1 = Rural_population_data
    Rural_population_data2 = Rural_population_data1.dropna() # drop null values from data.
    return Rural_population_data2

# call the function to get data
Rural_population_data3 = Rural_population_data2(Rural_population_data) 
Rural_population_data3.head(10) # shows starting rows from data.
# check shape of data.
Rural_population_data3.shape 
# check null values from data.
Rural_population_data3.isnull().sum()
#label encoder is used here, label encoder is used to encode categorical values, as the coutry name column contains categorical values
#therefore label encoder used here to encoder these values into numerical.
LE = LabelEncoder()
#taking names of country in a country namevariable
Country_Names = Rural_population_data3['Country Name']
#encoding the country name  column
Rural_population_data3['Country Name'] = LE.fit_transform(Rural_population_data3['Country Name']) 
Rural_population_data3.head(5)

#here minmax scaling is used, min max scaling is a technique which is used to scale the data in a range of 0 to 1, using minmax scaling the effect of outliers decreases.
#dropping indicator name column and make new dataframe X and y. X has all columns except country name and y has only names of countries

X = Rural_population_data3.drop(['Country Name','Indicator Name'], axis=1)
y = Rural_population_data3['Country Name']  

#initializing minmaxscaling, for minmaxscaling we fit the data to scale.
scale = MinMaxScaler()# define classifier.
Scaled_data = scale.fit_transform(X)# fit classifier with data.  
# Elbow Method
"""K means technique is used to make clusters of data. K here is the number of clusters but to decide value 
of k, elbow method is used. this method shows the average distance of clusters with different values of k.
TO decide best value, the elbow point is choosen as k."""

#making a list of clusters in range(0, 10)
K_values = range(10)
#creating a variable to store average distance of clusters
average_distance = list()

for K in K_values:
    #make cluster for value of k
    clustering = KMeans(n_clusters=K+1) 
    #fitting data
    clustering.fit(Scaled_data) 
    #calculating the average distance and appending to average distance variable. The distance of each data point to nearest cluster is calculated and then the mean of distances is appended to average distance
    average_distance.append(sum(np.min(cdist(Scaled_data, clustering.cluster_centers_, 'euclidean'), axis=1)) / Scaled_data.shape[0]) 

#here all the parameters for graph are set like title, label and size

# setting the font size here.
plots.rcParams.update({'font.size': 20})
# figure size for the plot
plots.figure(figsize=(10,7))
# a graph is plotted between average distance and k value
plots.plot(K_values, average_distance, marker="o", color = 'b') 
# adding x label to graph
plots.xlabel('<---------Numbers of Clusters---------->', color = 'r', fontsize = 15)
# adding x label to graph
plots.ylabel('Average distance', color = 'r', fontsize = 15) 
# adding title to graph
plots.title('Choosing k with the Elbow Method', color = 'green', fontsize = 20); 

# K-means Clustering

#As by using elbow method we take the value of k, the elbow point seen here is at k value as 3
#means 3 clusters are best for data

#creating 3 clusters for data
k_means_ = KMeans(n_clusters=3, max_iter=200, n_init=10,random_state=20)
# fitting the data to cluster
k_means_.fit(Scaled_data) 
#taking clustered data
clustered_data = k_means_.predict(Scaled_data)  
clustered_data 

# 3 clusters are made using kmeans clustering approach, 
# The number of points occuring in a cluster are shown here
cluster_label, cluster_points_count = np.unique(clustered_data, return_counts=True)

# Printing the count of Points occuring in each cluster
for value, count in zip(cluster_label, cluster_points_count):
    print(f"{value}: {count}")
# Scatter plot to show 3 clusters
# In this graph, a scatter plot for rural population data is shown between year 2000 and 2002
# define color for all clusters.
colors = {0 : 'r', 1 : 'b', 2 : 'g'} 

# this function return the color for cluster
def color_picker(x):  
    return colors[x]  

# creating a list that conatins unique cluster
cluster_color = list(map(color_picker, k_means_.labels_))   

#here all the parameters for graph are set like title, label and size

# setting the font size here.
plots.rcParams.update({'font.size': 20})
# figure size for the plot
plots.figure(figsize=(10,7))
# creating a scatter plot between year 2000 and year 2002.
plots.scatter(x=X.iloc[:,40], y=X.iloc[:,42], c=cluster_color)  
# adding x label to graph
plots.xlabel('<------2000------->', color = 'g', fontsize = 15)
# adding y label to graph
plots.ylabel('<------2002------->', color = 'g', fontsize = 15) 
# adding title to graph
plots.title('Scatter plot for 3 Clusters');  

#here the centroids of each cluster is calculated
centroids = k_means_.cluster_centers_
#unique labels are taken for each cluster
u_labels = np.unique(clustered_data) 
#showing the centroids data
centroids

# Scatter plot to show 3 clusters with thier centroids
# Here a scatter plot is shown with 3 clusters and their respective centroids
#setting the size for figure
plots.figure(figsize=(10,7))
for i in u_labels:
    plots.scatter(Scaled_data[clustered_data == i , 0] , Scaled_data[clustered_data == i , 1] , label = i)  

#here all the parameters for graph are set like title, label and size

# here a scatter plot is shown between year 2000 and year 2001, which shows 3 cluster and centroids
plots.scatter(centroids[:,40] , centroids[:,41] , s = 40, color = 'r') 
# adding x label to graph
plots.xlabel('<------2000------->', color = 'b', fontsize = 15)
# adding y label to graph
plots.ylabel('<------2001------->', color = 'b', fontsize = 15)
# adding title to graph
plots.title('Scatter plot for 3 Clusters with Centroids') 
#legends are added here
plots.legend()  
#showing the graph
plots.show()  

# creating lists that contains country names for each cluster.
# initializing thre clusters 
cluster1=[]
cluster2=[] 
cluster3=[] 

#making a k for loop, this for loop iterate through full cluster data and store country name for each cluster country in a variable
for i in range(len(clustered_data)):
    if clustered_data[i]==0:
        cluster1.append(Rural_population_data.loc[i]['Country Name']) 
    elif clustered_data[i]==1:
        cluster2.append(Rural_population_data.loc[i]['Country Name'])
    else:
        cluster3.append(Rural_population_data.loc[i]['Country Name'])
#converting first cluster data to numpy array
cluster_first_data = np.array(cluster1)
#showing the cluster data 
print(cluster_first_data)
# showing the data present in second cluster.
cluster_second_data = np.array(cluster2)
print(cluster_second_data)  

# showing the data present in third cluster.
cluster_third_data = np.array(cluster3)
print(cluster_third_data)  

# Here a country is selected from first cluster and population growth in that particular country from 1960 to 2021 is taken into a list
#taking a country from first cluster
first_cluster_country = cluster_first_data[3]
#Showing Country Name 
print('Country name :', first_cluster_country)

Africa_Western_and_Central_country_name = Country_Names[Country_Names == first_cluster_country]
country_indice = Africa_Western_and_Central_country_name.index.values

Africa_Western_and_Central_country = Rural_population_data3[Rural_population_data3['Country Name']==int(country_indice)]
Africa_Western_and_Central_country = np.array(Africa_Western_and_Central_country)  
Africa_Western_and_Central_country = np.delete(Africa_Western_and_Central_country, np.s_[:2]) 
Africa_Western_and_Central_country    

second_cluster_country = cluster_first_data[3] 
print('Country name :', second_cluster_country) 
Mexico_country_name = Country_Names[Country_Names == second_cluster_country]
country_indice = Mexico_country_name.index.values
Mexico_country = Rural_population_data3[Rural_population_data3['Country Name']==int(country_indice)] 
Mexico_country = np.array(Mexico_country)  
Mexico_country = np.delete(Mexico_country, np.s_[:2]) 
Mexico_country 

third_cluster_country = cluster_first_data[5] 
print('Country name :', third_cluster_country) 
India_country_name = Country_Names[Country_Names == third_cluster_country]
country_indice = India_country_name.index.values
India_country = Rural_population_data3[Rural_population_data3['Country Name']==int(country_indice)] 
India_country= np.array(India_country)  
India_country = np.delete(India_country, np.s_[:2]) 
India_country 
# Population growth of different countries from each cluster

#creating a list that contains years from 1960-2021
year=list(range(1960,2022))

#setting the figure size here
plots.figure(figsize=(22,8))

#Creating first subplot that shows the population growth of Africa Western and Central Country.
plots.subplot(131)
plots.xlabel('Years')
plots.ylabel('Population Growth') 
plots.title('Africa Western and Central Country') 
plots.plot(year,Africa_Western_and_Central_country, color='g');

#Creating second subplot that shows the population growth of Mexico Country.
plots.subplot(132)
plots.xlabel('Years')
plots.ylabel('Population Growth') 
plots.title('Mexico Country') 
plots.plot(year,Mexico_country);

#Creating third subplot that shows the population growth of India Country.
plots.subplot(133) 
plots.xlabel('Years') 
plots.ylabel('Population Growth')
plots.title('India Country') 
plots.plot(year,India_country, color='r');

# Curve Fitting
# Curve fit is shown here, curve is the graph that best fit to our points

#creating a new variable that contains rural population dataset
Rural_population_data_new = Rural_population_data.dropna()


# creating an array containing column names for dataset
Years_ = np.array(Rural_population_data_new.columns) 
#deleting 1st and 2nd column as they are country name and indicator name. We create a list that contains only years
Years_ = np.delete(Years_, 0) 
Years_ = np.delete(Years_, 0) 

#Converting years to integer type 
Years_ = Years_.astype(np.int)
print(Years_)

print('<------------------------------------------------------------>')
print('<------------------------------------------------------------>')
print('<------------------------------------------------------------>')

# selecting all the data for urban population and india.
India_country_population = Rural_population_data_new[(Rural_population_data_new['Indicator Name']=='Rural population') & (Rural_population_data_new['Country Name']=='India')]   

# convert into array.
India_country_population = India_country_population.to_numpy()
# dropping some columns.
India_country_population = np.delete(India_country_population, 0) 
India_country_population = np.delete(India_country_population, 0)
# convert data type as int.
India_country_population = India_country_population.astype(np.int) 
print('Population :', India_country_population)


# creating a function to plot a linear line
def linear_line(x, m, c):
    return m*x + c

def create_curve_fit(X_data,y_data): 

    # Perform curve fitting
    popt, pcov = curve_fit(linear_line, X_data, y_data) 

    # Extract the fitted parameters and their standard errors
    M, C = popt
    #taking square root of diagonals
    m_error, c_error = np.sqrt(np.diag(pcov)) 

    # Calculate the lower and upper limits of the confidence range
    #setting confidence to 95%
    conf_int = 0.95 
    alpha = 1.0 - conf_int 

    #cl=alculating upper and lower limit
    Low_m, High_m = scipy.stats.t.interval(alpha, len(X_data)-2, loc=M, scale=m_error)
    Low_c, High_c = scipy.stats.t.interval(alpha, len(X_data)-2, loc=C, scale=c_error)

    #plotting the best fitted graph
    #setting the figure size
    plots.figure(figsize=(12,6)) 
    #settign the font size as 20 here
    plots.rcParams.update({'font.size': 20})
    #plotting the data
    plots.plot(X_data, y_data, 'bo', label='Data', color = 'purple') #set data for graph.
    #plotting linear line
    plots.plot(X_data, linear_line(X_data, M, C), 'g', label='Fitted function')
    #showing confidence range
    plots.fill_between(X_data, linear_line(X_data, Low_m, Low_c), linear_line(X_data, High_m, High_c), color='gray', alpha=0.5, label='Confidence range') # set all the parameter.
    #settign title for graph
    plots.title('Curve Fitting', color = 'r', fontsize = 15) # define title for graph.
    #setting the x label 
    plots.xlabel('Years') 
    #setting the y label 
    plots.ylabel('<-------Population-------->', color = 'r', fontsize = 15) # define ylabel. 
    #settign legends
    plots.legend() 
    #showing the graph
    plots.show() 
    
# Curve fit is plotted here, All violet points are our data points
create_curve_fit(Years_, India_country_population)