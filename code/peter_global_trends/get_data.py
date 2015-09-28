### DATA IMPORT


### Modules

import numpy as np


### Reading the file

def retrieve (f):

    loading=open(f)

    content=loading.readlines()

    loading.close()


# No. of participants
    participants=content[0].count("\r")+1

# Splitting the rows
    rows=np.array([])
    
    for element in content:
        line=element.split("\r")
        rows=np.append(rows, line)

# Splitting the datapoints
    all_data=np.array([])
    
    for element in rows:
        datapoint=element.split(",")
        all_data=np.append(all_data, datapoint)

# Filling the participants into a data list
    data=[]
    
# No. of trials
    trials=len(all_data)/participants-2


    for x in range (1, participants):
        
        # An empty subject list
        subject=[]

        for y in range (x*(trials+2), (x+1)*(trials+2)):
            subject.append(all_data[y])
                
        data.append(subject)

    return data
