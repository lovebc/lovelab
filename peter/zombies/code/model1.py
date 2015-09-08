##### Model Specifications ######


### Modules
from __future__ import division
import numpy as np
import math

##### Model function #####
def model1_eval (mission, parameters, subject, coded_choices, division):
        #Unpack parameters
        constant=0
    
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
                        
			formula=constant
			
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