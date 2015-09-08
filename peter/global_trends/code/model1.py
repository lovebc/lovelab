##### Model Specifications ######


### Modules
from __future__ import division
import numpy as np
import math

##### Model function #####
def model1_eval (mission, parameters, subject, coded_choices, division):
        #Unpack parameters
        constant=parameters[0]

        # Initial uncertainty
        uncertainty=0
    
        # For estimation and prediction
        if mission[0]=="estimation" or mission[0]=="prediction":
            
                if mission[0]=="estimation":
                        a=0
                        b=division
                elif mission[0]=="prediction":
                        a=division
                        b=len(coded_choices)
        
                # Get value from mission input for prediction
                try:
                        uncertainty=mission[1]
                except:
                        uncertainty=0
           
                # Creating probabilities and values array
                probabilities=np.array([])
                    
                # Probabilities in each trial
                for i in range (a, b):
                        
			formula=0
			
			if i>0:
				if coded_choices[i-1]==0:
					formula=constant
				else:
					if coded_choices[i]==0:
                                                formula=-500
                                        elif coded_choices[i]==1:
                                                formula=500
			
                        # Limit the value so that exponent can be calculated
                        if formula>500:
                                formula=500
                        elif formula<-500:
                                formula=-500
                
                        prob=1/(1+math.exp(-(formula)))
                
                        # Probabilities
                        probabilities=np.append(probabilities, prob)
                
                return probabilities
        
        # To get the uncertainty value
        elif mission[0]=="get_values":
            
                a=0
                b=division
                
                for i in range (a, b):
            
			if i>0:
				uncertainty=0
            
                return uncertainty

##########