##### Model Specifications ######



### Modules

from __future__ import division
import numpy as np
import math


##### Model function #####
def model2_eval (mission, parameters, subject, coded_choices, division, coop_levels):
        #Unpack parameters
        slope=parameters[0]
	repeat=parameters[1]
    
        # For estimation and prediction
        if mission[0]=="estimation" or mission[0]=="prediction":
            
                if mission[0]=="estimation":
                        a=0
                        b=division
                elif mission[0]=="prediction":
                        a=division
                        b=len(coded_choices)
                    
                # Creating probabilities and values array
                probabilities=np.array([])
                    
                # Probabilities in each trial
                for i in range (a, b):
        
			formula=0
			
			if i>1:
				points_diff=0.25*float(coop_levels[i-1])*80
				diff_coop=float(coop_levels[i-1])-float(coop_levels[i-2])
				formula+=slope*float(coop_levels[i])
				if coded_choices[i-1]=="fighting":
					formula+=repeat
                    
                        # Limit the value so that exponent can be calculated
                        if formula>500:
                                formula=500
                        elif formula<-500:
                                formula=-500
                
                        prob=1/(1+math.exp(-(formula)))
                
                        # Probabilities
                        probabilities=np.append(probabilities, prob)
                
                return probabilities 

##########