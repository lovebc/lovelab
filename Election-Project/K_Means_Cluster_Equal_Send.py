# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 16:20:15 2014

@author: frankkanayet
"""

#K-Means Cluster
import os
import pandas as pd
import numpy as np
from sklearn import cluster
from sklearn.cluster import Ek_means_
#from sklearn.cluster import KMeans
#from sklearn.cluster import MiniBatchKMeans
#from scipy.spatial.distance import pdist
from scipy import *
import sys

#filepath = '/Users/frankkanayet/Documents/OSU/Research/Maps'
# This is the file of the table with the racial data for a single state.
# Later this will be a list of files one foe each state
infile = '../Maps/x_y_state_44_NAD83_FULL.csv'
#os.chdir(filepath)
#sys.path.append(filepath)


db = pd.read_csv(infile)
#print db['mx']
M = db.loc[:,['mx','my', 'ConDis']]
#db=[[-10,-10],[-11,-10],[-10,-11],[10,10],[10,11],[15,10]] #data to cluster
numClusters= max(M['ConDis'])
#clus = KMeans(n_clusters=numClusters, n_jobs = 1)	# initializes the model
clus = cluster.Ek_means_.KMeans(n_clusters=numClusters, n_jobs = -1, par = 1.2)
#clus=MiniBatchKMeans(n_clusters=2)	# initializes the model
clus.fit(M.loc[:,'mx':'my'])			# fits model to the data
labs = clus.labels_
print 'People per Cluster ', np.bincount(labs)

M['Labels'] = labs
peoplePerCluster = np.bincount(M['Labels'])
peoplePerDistrict = np.bincount(M['ConDis']-1)


M.to_csv('Clustered_44.csv')

print 'File saved:\n'

print 'Starting measure of distance for cluster solution:\n'

results_Cluster = {}
results_State = {}
totalDist = np.zeros(numClusters, double)
for i in range(numClusters):
    print 'Cluster_',i 
    data = M.loc[M['Labels']==i,'mx':'my']
    for j in xrange(peoplePerCluster[i]-1):
        if j % 10000 == 0:
            print "%s/%s (%0.2f%%)"%(j+1,peoplePerCluster[i],100*((j+1)/float(peoplePerCluster[i])))
        totalDist[i]+=sum(sqrt(add.reduce(((data[(j+1):peoplePerCluster[i]]-data.iloc[j])**2),1)))
    results_Cluster['Avg_Dist_Cluster_'+str(i)] = [totalDist[i]/(peoplePerCluster[i]*(peoplePerCluster[i]-1)/2.)]
    results_Cluster['Num_People_Cluster_'+str(i)] = [peoplePerCluster[i]]
avgDist = totalDist/(peoplePerCluster*(peoplePerCluster-1)/2.)
avgDistForState=inner(avgDist,peoplePerCluster)/sum(peoplePerCluster)    
results_State['Cluster_Avg_Distance'] = [avgDistForState] 
print 'AvgDist', avgDist

res_cluster = pd.DataFrame(results_Cluster)
res_state = pd.DataFrame(results_State)
res_cluster.to_csv('Cluster_results.csv')
res_state.to_csv('State_results.csv')

print 'Results files saved temporarily:\n'
#Now the same analysis for real congressional districts

totalDist_District = np.zeros(numClusters, double)
for i in range(numClusters):
    print 'District_',i
    data_Dis = M.loc[M['ConDis']==i+1,'mx':'my']
    for j in xrange(peoplePerDistrict[i]-1):
        if j % 100000 == 0:
            print "%s/%s (%0.2f%%)"%(j+1,peoplePerDistrict[i],100*((j+1)/float(peoplePerDistrict[i])))
        totalDist_District[i]+=sum(sqrt(add.reduce(((data_Dis[(j+1):peoplePerDistrict[i]]-data_Dis.iloc[j])**2),1)))
    results_Cluster['Avg_Dist_District_'+str(i+1)] = [totalDist_District[i]/(peoplePerDistrict[i]*(peoplePerDistrict[i]-1)/2.)]
    results_Cluster['Num_People_District_'+str(i+1)] = [peoplePerDistrict[i]]
avgDist = totalDist_District/(peoplePerDistrict*(peoplePerDistrict-1)/2.)
avgDistForState=inner(avgDist,peoplePerDistrict)/sum(peoplePerDistrict)    
results_State['District_Avg_Distance'] = [avgDistForState] 
print 'AvgDist', avgDist

res_cluster = pd.DataFrame(results_Cluster)
res_state = pd.DataFrame(results_State)
res_cluster.to_csv('Cluster_results.csv')
res_state.to_csv('State_results.csv')

print 'District results files saved:\n'

print 'Now make plot of new map:\n'

plt.scatter(M['mx'],M['my'], c = labs)
plt.savefig('Cluster_map_44.png')

print 'Plot saved... We are done!'


###Following code takes too long...

#for i in range(numClusters):
#    print 'Cluster_',i 
#    data = M.loc[M['Labels']==i,'mx':'my']
#    for j in xrange(len(data)):
#        if j % 1000 == 0:
#            print 'First loop:\n'
#            print "%s/%s (%0.2f%%)"%(j+1,peoplePerCluster[i],100*((j+1)/float(peoplePerCluster[i])))
#        for k in xrange(j+1,len(data)):
#            if k % 100000 == 0:
#                print 'Second loop:\n'
#                print "%s/%s (%0.2f%%)"%(k+1,len(range(j+1,len(data))),100*((k+1)/float(len(range(j+1,len(data))))))
#            totalDist[i] += sqrt(sum((data.iloc[j]-data.iloc[k])**2))
#    results_Cluster['Avg_Dist_Cluster_'+str(i)] = totalDist[i]/(peoplePerCluster*(peoplePerCluster-1)/2.)
#    results_Cluster['Num_People_Cluster_'+str(i)] = peoplePerCluster[i]
#
#avgDist=totalDist/(peoplePerCluster*(peoplePerCluster-1)/2.)  #average distance in a person in a district to every other person in district. 
#avgDistForState=inner(avgDist,peoplePerCluster)/sum(peoplePerCluster)
#results_State['Cluster_Avg_Distance'] = avgDistForState 
#
#res_cluster = pd.DataFrame(results_Cluster)
#res_state = pd.DataFrame(results_State)
#res_cluster.to_csv('Cluster_results.csv')
#res_state.to_csv('State_results.csv')
#
#print 'Results files saved temporarily:\n'
##Now the same analysis for real congressional districts
#
#print 'Now the same analysis for real congressional districts:\n'
#
#totalDist_District = np.zeros(numClusters, double)
#for i in range(numClusters):
#    print 'District_', i
#    data_Dis = M.loc[M['ConDis']==i+1,'mx':'my']
#    for j in xrange(len(data_Dis)):
#        if j % 1000 == 0:
#            print 'First loop:\n'
#            print "%s/%s (%0.2f%%)"%(j+1,peoplePerDistrict[i],100*((j+1)/float(peoplePerDistrict[i])))
#        for k in xrange(j+1,len(data_Dis)):
#            if k % 100000 == 0:
#                print 'Second loop:\n'
#                print "%s/%s (%0.2f%%)"%(k+1,len(range(j+1,len(data))),100*((k+1)/float(len(range(j+1,len(data))))))
#            totalDist_District[i] += sqrt(sum((data_Dis.iloc[j]-data_Dis.iloc[k])**2))
#    results_Cluster['Avg_Dist_District_'+str(i)] = totalDist_District[i]/(peoplePerDistrict*(peoplePerDistrict-1)/2.)
#    results_Cluster['Num_People_District_'+str(i)] = peoplePerDistrict[i]
#
#avgDist=totalDist_District/(peoplePerDistrict*(peoplePerDistrict-1)/2.)  #average distance in a person in a district to every other person in district. 
#avgDistForState=inner(avgDist,peoplePerDistrict)/sum(peoplePerDistrict)
#results_State['District_Avg_Distance'] = avgDistForState 
#
#res_cluster = pd.DataFrame(results_Cluster)
#res_state = pd.DataFrame(results_State)
#res_cluster.to_csv('Cluster_results.csv')
#res_state.to_csv('State_results.csv')
#
#plt.scatter(M['mx'],M['my'], c = labs)
#plt.savefig('Cluster_map_44.png')


### And this code has memory errors

#for i in range(numClusters):
#    distM=pdist(M[M['Labels']==i])
#    #print(around(distM,2))
#    results.append(sum(distM)/(len(distM))) #average distance between people in a district
#    #peoplePerDistict.append(len(distM)) #number of people in the district
#    results_Cluster['Avg_Dist_Cluster_'+str(i)] = sum(distM)/(len(distM))
#    results_Cluster['Num_People_Cluster_'+str(i)] = peoplePerCluster[i]
#
#print(results,peoplePerCLuster)    
#print("average for the entire state: ",inner(results,peoplePerCluster)/sum(peoplePerCluster)) #I think this does a weighted average for the entire state)
#results_State['Cluster_Avg_Distance'] = inner(results,peoplePerCluster)/sum(peoplePerCluster)
#
##Now the same analysis for real congressional districts
#results_Dis=[]
##peoplePerDistict=[]
#for i in range(numClusters):
#    distM_Dis=pdist(M[M['ConDis']==i+1])
#    #print(around(distM,2))
#    results_Dis.append(sum(distM_Dis)/(len(distM_Dis))) #average distance between people in a district
#    #peoplePerDistict.append(len(distM)) #number of people in the district
#    results_District['Avg_Dist_District_'+str(i)] = sum(distM_Dis)/(len(distM_Dis))
#    results_District['Num_People_District_'+str(i)] = peoplePerDistrict[i]
#
#print(results_Dis,peoplePerDistrict)    
#print("average for the entire state: ",inner(results_Dis,peoplePerDistrics)/sum(peoplePerDistrict)) #I think this does a weighted average for the entire state)
#results_State['District_Avg_Distance'] = inner(results_Dis,peoplePerDistrict)/sum(peoplePerDistrict)
#
