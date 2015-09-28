##### Model estimation script #####

### Modules ###
from __future__ import division
import math
import random
import numpy as np
import multiprocessing
import subprocess
import itertools
from scipy.optimize import minimize
from randomize import randomize_choices
import gc
# Import data
from get_data import retrieve
dataset=retrieve('data.csv')

# Models for fitting and prediction
from model1 import *
from model2 import *

#Disable garbage collection
gc.disable()

##########

### Settings ###

comparisons=100 # How many randomized versions to compare?
cross_validate=0.20 # How many datapoints are being predicted? 0.00 (just estimation) to 1.00 (all)

# Model parameter configurations
tested_models=['1','2'] # Models to be tested by indicator number
parameters={'1':["constant"], '2':["constant", "streak_par"]}
initials={'1':[0.00000], '2':[0.00000, 0.00000]}
bounds={'1':[[-49.99999,49.99999]], '2':[[-49.99999,49.99999],[-9.99999,9.99999]]}
grid_size=1 # Takes (grid_size*2+1) probes from within the bounds of each parameter (the center + grid_size*probes from each side of the center)
### Number of cores

no_cores=None #All=None
##########


##### Create list of initial parameter settings to be tested #####

initial_parameters={}
for m in tested_models:
	initial_parameters[m]=[]
N=grid_size

for m in tested_models:

	all_inits=[] # all potential values for all parameters (unordered)
	par_set=[] # all potential values as set of parameters
	
	no_parameters=len(parameters[m]) # number of parameters
	bounds_model=bounds[m] # parameter bounds
	initials_model=initials[m] # initial parameter values
	
	for g in range(0, grid_size):
		initial_parameters[m].append(initials_model)
	
	for par in range(0, no_parameters):
		this_parameter=[] # all potential values for one parameter
		center=(bounds_model[par][1]+bounds_model[par][0])/2
		distance=bounds_model[par][1]-center
		sampling_interval=distance/(N+2)
	
		for k in range(-N, N+1):
			temp=center+k*sampling_interval
			this_parameter.append(temp)
		all_inits.append(this_parameter)
	
	for indices in itertools.product(range(2*N+1), repeat=no_parameters):
		for i in range(0, len(indices)):
			par_set.append(all_inits[i][indices[i]])
		initial_parameters[m].append(par_set)
		par_set=[]

##########


##### Model fit function #####

def get_fit (x, y, subject, coded_choices, data_cut, r_streaks, mt, double_check):

	### Starting parameters and bounds

	init_pars=initial_parameters[mt]
	bounds_model=bounds[mt]

	#####

	### LogLikelihood function

	def loglike (par, y, x, subject, coded_choices, division, r_streaks, mt):
		Max_LL=0
		if mt=='1':
			p_values=model1_eval(["estimation"], par, subject, coded_choices, division)
		elif mt=='2':
			p_values=model2_eval(["estimation"], par, subject, coded_choices, division, r_streaks)
			
		for value in range (0, len(y)):
			if y[value]==0:
				Max_LL-=math.log(1-p_values[value])
			elif y[value]==1:
				Max_LL-=math.log(p_values[value])
		return Max_LL

	#####

	### Optimizing fit-values

	# Empty optimal-list for parameters and initial dummy value for the LL-function
	opti=[]
	value=-99999

	try:
		# Minimizing the Loglike-function with SLSQP
		for initial in init_pars:
			try:
				temp = minimize(loglike, initial, args=(y, x, subject, coded_choices, data_cut, r_streaks, mt),
						bounds=bounds_model, method='SLSQP', options={'disp': False})

				temp_pars=temp.x[0:no_parameters]

				if -temp.fun>value:
					value=-temp.fun
					opti=temp_pars
			except:
				pass
			
			if value!=-99999 and len(opti)!=0:
				break

		# If failed to optimize with SLSQP, try TNC
		if value==-99999 and len(opti)==0 and double_check==True:
			for initial in init_pars:
				try:
					temp = minimize(loglike, initial, args=(y, x, subject, coded_choices, data_cut, r_streaks, mt),
							bounds=bounds_model, method='TNC', options={'disp': False})
	
					temp_pars=temp.x[0:no_parameters]
	
					if -temp.fun>value:
						value=-temp.fun
						opti=temp_pars
				except:
					pass
				
				if value!=-99999 and len(opti)!=0:
					break

	except:
		pass
	
	# Set parameters back if fitting failed
	if value==-99999:
		opti=initials[mt]
		
	#####
	
	return [value, opti]

##########


##### Optimization function #####
def optimizing (participant):	
	try:
		
		##### Subject #####
		
		# Creating a subject var that is an integer
		subject=int(participant)		
		### Subject's data
		
		subj_data=dataset[subject]
		choices=subj_data[2:len(subj_data)]
		condition=subj_data[1]
		name=subj_data[0]
		
		# Clean data and cut-off initial trials
		
		coded_choices=[]
		begin=False
		for c in choices:
			this_choice=int(c)
			if this_choice!=-99:
				begin=True
			if begin==True:
				coded_choices.append(this_choice)
		##########


		##### SHUFFLED choices #####
		
		shuffled_coded=[]
		for c in range(0, comparisons):
			shuffled_coded.append(randomize_choices(coded_choices))

		##########


		##### OBSERVED Streak lengths #####
	
		r_streaks=[] # Streak lengths since last exploration
		rsl=0
	
		for g in coded_choices:
			if g==1:
				rsl=0
			else:
				rsl+=1
			r_streaks.append(rsl)
	
		##########
		
		
		##### SHUFFLED Streak lengths #####
		
		shuffled_streaks=[]
	
		for c in range(0, comparisons):
			these_r_streaks=[] # Streak lengths of repetitions/exploitations
			rsl=0
		
			for g in shuffled_coded[c]:
				if g==1:
					rsl=0
				else:
					rsl+=1
				these_r_streaks.append(rsl)
			shuffled_streaks.append(these_r_streaks)
	
		##########


		##### Define fitted data part for cross-validation #####
		
		fitted_data=int(round((1-cross_validate)*len(coded_choices)))
		
		##########
		
		##### OBSERVED Estimation function inputs (x, y) #####
		# X-values (array of 1s)
		x_est = np.array([])
		x_fit = np.array([])

		for i in range (0, len(coded_choices)):
			x_est = np.append(x_est, 1)
		for i in range (0, fitted_data):
			x_fit = np.append(x_fit, 1)
		# Y-values (binominal choice matrix with 0=Exploit, 1=Explore)
		y_est=np.array([])
		y_fit=np.array([])
		y_pred=np.array([])
		
		for g in range (0, len(coded_choices)):
			if coded_choices[g]==0:
				y_est=np.append(y_est, 0)
			elif coded_choices[g]==1:
				y_est=np.append(y_est, 1)
			else:
				y_est=np.append(y_est, -99)
		for g in range (0, fitted_data):
			if coded_choices[g]==0:
				y_fit=np.append(y_fit, 0)
			elif coded_choices[g]==1:
				y_fit=np.append(y_fit, 1)
			else:
				y_fit=np.append(y_fit, -99)
		for g in range (fitted_data, len(coded_choices)):
			if coded_choices[g]==0:
				y_pred=np.append(y_pred, 0)
			elif coded_choices[g]==1:
				y_pred=np.append(y_pred, 1)
			else:
				y_pred=np.append(y_pred, -99)
		##########
		
		
		##### SHUFFLED Estimation function inputs (x, y) #####
		
		shuffled_x=[]
		shuffled_y=[]
		
		for c in range(0, comparisons):
			
			# X-values (array of 1s)
			x_values = np.array([])
			for i in range (0, len(shuffled_coded[c])):
				x_est = np.append(x_est, 1)
	
			# Y-values (binominal choice matrix with 0=Exploit, 1=Explore)
			y_values=np.array([])
			for g in range (0, len(shuffled_coded[c])):
				if shuffled_coded[c][g]==0:
					y_values=np.append(y_values, 0)
				elif shuffled_coded[c][g]==1:
					y_values=np.append(y_values, 1)
				else:
					y_values=np.append(y_values, -99)

			shuffled_x.append(x_values)
			shuffled_y.append(y_values)

		##########


		##### Get performances for the different models #####
		
		# Empty results dictionary for models
		results={'id':name, 'condition':condition, 'estimated':-99, 'predicted':-99}
		
		# Store LL-values for fit and predictions
		performances={}
		for model_type in tested_models:
			performances[model_type]={}
	
		for model_type in tested_models:

			# Empty results slot
			results[model_type]=[]
			
			### Get fits and parameter-values
			
			all_data_estimation=get_fit(x_est, y_est, subject, coded_choices, len(coded_choices), r_streaks, model_type, True)
			part_data_estimation=get_fit(x_fit, y_fit, subject, coded_choices, fitted_data, r_streaks, model_type, True)
			overall_fit=all_data_estimation[0]
			overall_parameters=all_data_estimation[1]
			prediction_parameters=part_data_estimation[1]
			
			#####
			
			### Prediction function
			
			if cross_validate>0:

				try:
					### Additional values from optimization
					
					try:
						if model_type=='2':
							uncertainty=model2_eval(["get_values"], prediction_parameters, subject, coded_choices, fitted_data, r_streaks)
					except:
						uncertainty=0
						
					#####
					
					### Get prediction performance
			
					cv_result=0
					
					if model_type=='1':
						pred_p_values=model1_eval(["prediction"], prediction_parameters, subject, coded_choices, fitted_data)
					elif model_type=='2':
						pred_p_values=model2_eval(["prediction", uncertainty], prediction_parameters, subject, coded_choices, fitted_data, r_streaks)
						
					for v in range (0, len(y_pred)):
						if y_pred[v]==0:
							cv_result+=math.log(1-pred_p_values[v])
						elif y_pred[v]==1:
							cv_result+=math.log(pred_p_values[v])
					
					#####
					
				except:
					print "Error predicting subject "+str(subject)
					cv_result=-99999
					pass
				
			else:
				cv_result=-99999
			
			##########
			
			
			##### Adding values into respective lists/arrays #####
			
			# Performance comparison
			performances[model_type]['overall_fit']=all_data_estimation[0]
			performances[model_type]['part_fit']=part_data_estimation[0]
			performances[model_type]['prediction']=cv_result
	

			# Formatting the results to 5-digit float numbers
			formatted_value="{0:.5f}".format(overall_fit)
			formatted_cv="{0:.5f}".format(cv_result)
			formatted_opti=[]
			for parameter in range(0,len(overall_parameters)):
				formatted_par="{0:.5f}".format(overall_parameters[parameter])
				formatted_opti.append(formatted_par)

			results["estimated"]=len(coded_choices)
			results["predicted"]=len(coded_choices)-fitted_data
			results[model_type]=[formatted_value, formatted_cv, formatted_opti]		
			##########
			
		
		##### Create significance indicators #####
		
		overall_chi_square=2*(performances['2']['overall_fit']-performances['1']['overall_fit'])
		part_chi_square=2*(performances['2']['part_fit']-performances['1']['part_fit'])
		predictive_performance=performances['2']['prediction']-performances['1']['prediction']
		overall_sign=0
		part_sign=0
		better_pred=0
		
		if overall_chi_square>3.84:
			overall_sign=1
		if part_chi_square>3.84:
			part_sign=1
		if predictive_performance>0:
			better_pred=1
		elif predictive_performance<0:
			better_pred=-1
			
		results['overall_sign']=overall_sign
		results['part_sign']=part_sign
		results['predictive_performance']=better_pred
		
		##########
			
		
		##### Get rank comparison of the slope #####
		
		try:	
			shuffled_slopes=[]
			for c in range (0, comparisons):
				this_comparison=get_fit(shuffled_x[c], shuffled_y[c], subject, shuffled_coded[c], len(shuffled_coded[c]), shuffled_streaks[c], '2', False)
				if this_comparison[0]!=-99999:
					shuffled_slopes.append(this_comparison[1][1])
				
			rank=0
			shuffled_slopes.sort()
			obs_slope=float(results['2'][2][1])
			for s in shuffled_slopes:
				if s<obs_slope:
					rank+=1
				elif s==obs_slope:
					rank+=0.5
			percentile=str("{0:.3f}".format(rank/len(shuffled_slopes)))
		except:
			percentile=-99
			pass
		results['percentile']=percentile
	except Exception, incident:
		print "Woops for subject: "+str(subject)+" with reason: "+str(incident)
		results={'id':subject, 'online':-99, 'estimated':-99, 'predicted':-99, 'overall_sign':-99, 'part_sign':-99, 'predictive_performance':-99, 'percentile':-99}
		for m in tested_models:
			results[m]=[-99999,-99999,initials[m]]
		pass
	##########
	
	finally:
		
		### Process so far
	    
		try:
			done=open("process_models.csv", "r+")
			content=done.readlines()
		
		    # Reading the rows
			count=0
			for element in content:
				count+=1
		
			critical=0
			i=1
			while critical<count:	
				critical=int(0.05*i*len(subjects))
				i+=1
	    
			process=100*count/len(subjects)
			process_int=int(round(process,0))
			process_str=str("{0:.0f}".format(process))+"%"
			done.write(process_str+"\n")
		
			if critical==count:
				print "Process: "+str(i*5)+"%"
		
			done.close()    
	    
		except:
			done=open("process_models.csv", "w+")
			done.write("0%\n") 
			done.close()
	    
		##########

	return results##########


####################


##### Writing the headlines for the results file #####

storage=open("results_models.csv", "w")

storage.write("Subject,Condition")

storage.write(",fitted,predicted")

for m in tested_models:
	storage.write(",LL_fit_"+m+",LL_pred_"+m)

	for parameter in parameters[m]:
		storage.write(","+parameter)
		
storage.write(",slope_perc")
storage.write(",sign_all")
storage.write(",sign_part")
storage.write(",pred")

##########


##### Obtaining the model results for every category and appending them to dictionary #####
	
# Status
print "\n --- Retrieving model results --- \n"

# Reset process file
done=open("process_models.csv", "w+")
done.close()

# Number of subjects
no_subjects=len(dataset)

    
# Making a list of the subjects (numbers from 0 on) to divide them into tasks
subjects=np.array([])
for participant in range (0, no_subjects):
	subjects=np.append(subjects, participant)


# Splitting function

def central(tasks):
	pool=multiprocessing.Pool(no_cores)
	results=[]
	print "\nStarting."
	print "\nTotal number of subjects: "+str(len(tasks))+"\n"
	r=pool.map_async(optimizing, tasks, callback=results.append)
	r.wait()
	return results

def main(tasks):
	output=central(tasks)
	return output

# Results array
Overall_Results=main(subjects)[0]

# Print 100% process
print "\nFinished."

# Sort results into all_results array subject-wise
for s in range(0, no_subjects):
	storage.write("\n"+str(Overall_Results[s]['id']))
	storage.write(","+str(Overall_Results[s]['condition']))
	storage.write(","+str(Overall_Results[s]['estimated']))
	storage.write(","+str(Overall_Results[s]['predicted']))
	
	for m in tested_models:
		# LL Fit and Prediction
		for i in range(0, 2):
			try:
				storage.write(","+str(Overall_Results[s][m][i])) # LL Fit (0) and LL Pred (1)
			except:
				storage.write(",-99999")
		# Parameter values
		for p in range(0, len(parameters[m])):
			try:
				storage.write(","+str(Overall_Results[s][m][2][p]))
			except:
				storage.write(",-99")
				
	storage.write(","+str(Overall_Results[s]['percentile']))
	storage.write(","+str(Overall_Results[s]['overall_sign']))
	storage.write(","+str(Overall_Results[s]['part_sign']))
	storage.write(","+str(Overall_Results[s]['predictive_performance']))

storage.close()##########


print "\nAll done."