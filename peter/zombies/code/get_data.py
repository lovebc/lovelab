### DATA IMPORT


### Modules

import numpy as np


### Reading the file

def retrieve (f):

    loading=open(f)

    content=loading.readlines()

    loading.close()


# No. of participants
    participants=content[0].count("\r")

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
    trials=(len(all_data)/(participants+1)-2)/3

    for x in range (1, participants+1):
        
        # An empty subject list
        subject=[]
        choices=[]
        fighting=[]

        start=x*(3*trials+2)
        end=(x+1)*(3*trials+2)

        for y in range (start,end):
            if y==(start):
                subject.append(all_data[y])
            elif y>start and y<=start+trials:
                choices.append(all_data[y])
            elif y>start+trials and y<=start+2*trials:
                fighting.append(all_data[y])

        subject.append(choices)
        subject.append(fighting)
        data.append(subject)

    return data
