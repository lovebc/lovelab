# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 16:20:15 2014

@author: frankkanayet
"""

#K-Means Cluster
import os
import multiprocessing
import subprocess
from time import time
#from scipy.spatial.distance import pdist
import pandas as pd
import numpy as np
from sklearn import cluster
from sklearn.cluster import Ek_means_
#from sklearn.cluster import KMeans
#from sklearn.cluster import MiniBatchKMeans
#from scipy.spatial.distance import pdist
from scipy import *
import sys

################################################################################	
# Parallel distance functions
# dataTask is the block of coordinates to work on and conditions for loop (start,end,step_size) where step_size 1 processes every item and step size size 2 every other, etc.

def ByClusterDistParallelHelper(dataTask):
	dataSegment=dataTask[0]
	helperDist=0
	print("dataTask",dataTask[1:4])
	peopleInCluster=len(dataSegment)
	for j in xrange(dataTask[1],dataTask[2],dataTask[3]):
		helperDist+=sum(np.sqrt(np.add.reduce((dataSegment.iloc[(j+1):peopleInCluster]-dataSegment.iloc[j])**2,1)))
	return helperDist

#############################
# This function should be ready for prime time use in project
### what is the structure of data??? data should be called with M	
def ByClusterDistParallel(data,step_size,label_name):
    #some defintions
    pool = multiprocessing.Pool(30) #this defines the pool of processors/cores. the parameter says how many processes to use. "None" defaults to number defined by the system. on my mac, it also counts virtual cores (hyperthreading).
    print("number of processes: ",pool._processes)
    ######### Need to change data.num for appropriate varaible name for number of clusters in principle is numClusters
    peoplePerCluster=np.zeros(numClusters,int) #number of people in each cluster
    totalDist=0
    ############### Same here numClusters
    totalDistPerCluster=np.zeros(numClusters,float) #sum of pairwise distances for each cluster
    ################ same in next line... numClusters
    for i in xrange(numClusters): # loop by cluster
        print("Starting Cluster #",i)
        ################ data.people needs to be changed by the list of xy coords
        ################ I actually have a way to do this with pandas
        temp = data.loc[data[label_name]==i,'mx':'my']
        temp=temp.iloc[range(0,len(temp),step_size)] #**** try shrinking the data matrix here by just discarding some data.
        peoplePerCluster[i]=len(temp) ## effective peoperPerCLuster after trimming
        #############    	create split of tasks for multiprocessing.
        tasks=[]
        taskSize=peoplePerCluster[i]/pool._processes
        for j in xrange(pool._processes):
            tasks.append([temp,j*taskSize,(j+1)*taskSize,1]) #step_size removed.
        tasks[j][2]=peoplePerCluster[i]-1 	#this ensures the last task gets any extra people resulting from dividing by pool.processes
        ############	
        results = []
        r = pool.map_async(ByClusterDistParallelHelper, tasks, callback=results.append) #splits up the tasks across processors.
        r.wait() # Wait on the results
        totalDist+=np.sum(results) #this will be removed.
        totalDistPerCluster[i]=np.sum(results)
    avgDistPerCluster=(totalDistPerCluster)/(peoplePerCluster*(peoplePerCluster-1)/2.)  #step_size no longer in numerator
    print("NOTE: STEP_SIZE REDUCES EFFECTIVE COUNTS:",step_size)
    print("Average Distance for each cluster:",avgDistPerCluster)
    print("People per cluster:",peoplePerCluster, "Real people per cluster:", RealpeoplePerCluster)
    print("Overall Average Distance:",np.inner(avgDistPerCluster,peoplePerCluster)/np.sum(peoplePerCluster))
    return avgDistPerCluster, peoplePerCluster

################################################################################

def main():
    print("START!")
    results_Cluster = {}
    results_State = {}
    results_Cluster['State'] = state
    results_State['State'] = state
    start_time=time()
    step_size = 1
    avgDistPerCluster, peoplePerCluster = ByClusterDistParallel(M,step_size,'Labels')		#solve using parallel method with step size 10 
    print("execution time: ",time()-start_time)
    for i in range(numClusters):
        results_Cluster['Avg_Dist_Cluster_'+str(i)] = avgDistPerCluster[i]
        results_Cluster['Num_People_Cluster_'+str(i)] = [peoplePerCluster[i]]
        results_Cluster['Real_Num_People_Cluster_'+str(i)] = [RealpeoplePerCluster[i]]
    avgDistForState=np.inner(avgDistPerCluster,peoplePerCluster)/np.sum(peoplePerCluster)
    results_State['Cluster_Avg_Distance'] = [avgDistForState]
    results_Cluster['Step_Size'] = step_size
    res_cluster = pd.DataFrame(results_Cluster)
    res_state = pd.DataFrame(results_State)
    res_cluster.to_csv('/scratch3/Frank_Files/Cluster_results.csv')
    res_state.to_csv('/scratch3/Frank_Files/State_results.csv')
    print 'Results files saved temporarily:\n'
    print 'Now do results for actual congressional districts:\n'
    start_time=time()
    avgDistPerDistrict, peoplePerCluster = ByClusterDistParallel(M,step_size,'ConDis')		#solve using parallel method with step size 10 
    print("execution time: ",time()-start_time)
    for i in range(numClusters):
        results_Cluster['Avg_Dist_District_'+str(i)] = avgDistPerDistrict[i]
        results_Cluster['Real_Num_People_District_'+str(i)] = [RealpeoplePerDistrict[i]]
        results_Cluster['Num_People_District_'+str(i)] = [peoplePerCluster[i]]
    avgDistForState=np.inner(avgDistPerDistrict,peoplePerCluster)/np.sum(peoplePerCluster)
    results_State['District_Avg_Distance'] = [avgDistForState]
    print("execution time: ",time()-start_time)
    return results_Cluster, results_State
###########################################################

# let's start
###########################################################
################################################
#filepath = '/Users/frankkanayet/Desktop/Maps'
#os.chdir(filepath)
#infile = 'x_y_state_44_NAD83_FULL.csv'
# This is the file of the table with the racial data for a single state.
# Later this will be a list of files one foe each state
#state = ['1', '6', '15', '32', '44']
state = ['6']
state_dict = {'1':'AL','2':'AK','4':'AZ','5':'AR','6':'CA','8':'CO','10':'DE',
              '11':'DC','12':'FL','13':'GA','15':'HI','16':'ID','17':'IL','18':'IN',
              '19':'IA','20':'KS','21':'KY','22':'LA','23':'ME','24':'MD','25':'MA',
              '26':'MI','27':'MN','28':'MS','29':'MO','30':'MT','31':'NE','32':'NV',
              '33':'NH','34':'NJ','35':'NM','36':'NY','37':'NC','38':'ND','39':'OH',
              '40':'OK','41':'OR','42':'PA','44':'RI','45':'SC','46':'SD','47':'TN',
              '48':'TX','49':'UT','50':'VT','51':'VA','53':'WA','54':'WV','55':'WI',
              '56':'WY'}
for st in state:
    print 'state:', st
    infile = '/scratch3/Frank_Files/x_y_state_'+st+'_NAD83_FULL.csv'
    #infile = '../Maps/x_y_state_44_NAD83_FULL.csv'
    #sys.path.append(filepath)
    Clus_start_time=time()
    db = pd.read_csv(infile)
    #print db['mx']
    M = db.loc[:,['mx','my', 'ConDis']]
    del db
    #db=[[-10,-10],[-11,-10],[-10,-11],[10,10],[10,11],[15,10]] #data to cluster
    numClusters= max(M['ConDis'])
    test = False
    par = 0
    while test == False:
        test1 = False
        ctr = 0
        init_centers = 'random'
        while test1 == False:
            print 'par ', par
            print 'ctr ', ctr
            #print 'init_centers', init_centers
            clus = cluster.Ek_means_.KMeans(n_clusters=numClusters, max_iter= 400, n_init= 1, init= init_centers, n_jobs = -1, par = par)
            clus.fit(M.loc[:,'mx':'my'])			# fits model to the data
            labs = clus.labels_
            pplPerCluster = np.bincount(labs)
            print 'pplPerCluster ', pplPerCluster
            ClusMax = max(pplPerCluster)
            ClusMin = min(pplPerCluster)
            print 'ClusMax', ClusMax, 'ClusMin', ClusMin
            if ClusMax <= 1100000 and ClusMin >= 500000: 
                test1 = True
            if ctr >= 10:
                test1 = True
            ctr +=1
            #init_centers = clus.cluster_centers_
            #pplPerClusterList = pplPerCluster.tolist()
            #indMax = pplPerClusterList.index(ClusMax)
            #indMin = pplPerClusterList.index(ClusMin)
            #init_centers[indMin] = init_centers[indMax]
        if ClusMax <= 1100000 and ClusMin >= 500000: 
            break 
        if par >= 2 and ctr >= 10:
            print 'Could not find a good solution'
            M['Labels'] = 'No solution' ## Maybe include this message in results so I know...
            test = True
        par += 0.2
    print("Cluster execution time: ",time()-Clus_start_time)
    M['Labels'] = labs
    RealpeoplePerCluster = np.bincount(M['Labels'])
    RealpeoplePerDistrict = np.bincount(M['ConDis'])
    print 'People per Cluster ', RealpeoplePerCluster
    print 'People per District ', RealpeoplePerDistrict
    M.to_csv('/scratch3/Frank_Files/Clustered_'+st+'.csv')
    print 'File saved:\n'
    print 'Starting measure of distance for cluster solution:\n'
    if __name__ == '__main__':
        results_Cluster, results_State = main()
    res_cluster = pd.DataFrame(results_Cluster)
    res_state = pd.DataFrame(results_State)
    res_cluster.to_csv('/scratch3/Frank_Files/Cluster_results.csv')
    res_state.to_csv('/scratch3/Frank_Files/State_results.csv')
    print 'District results files saved:\n'
#print 'Now make plot of new map:\n'

#plt.scatter(M['mx'],M['my'], c = labs)
#plt.savefig('Cluster_map_44.png')

#print 'Plot saved... We are done!'


