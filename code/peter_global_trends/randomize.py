##### Randomizing choices #####


##### Modules #####

import random

##########

def randomize_choices (choices):

	# Lists
	index_list_choices=[]
	rand_choices=[]
	
	# Choices made
	for x in range(0, len(choices)):
		index_list_choices.append(x)
	    
	for y in range(0, len(choices)):
		a=random.randint(0, (len(index_list_choices)-1))
		b=index_list_choices[a]
		rand_choices.append(choices[b])
		del index_list_choices[a]
	
	return rand_choices

##########