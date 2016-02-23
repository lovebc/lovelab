So the files that we have work like this:

For every state the information is obtained from 3 different files:

In all files 'n' equals state number...

1. Clustered_n.csv
This file has the results of our clustering method and are compared to the original congressional districts.
	mx	my	ConDis	Labels
0	823716.0154	-835647.7173	1	3
1	822875.7228	-835862.3188	1	3
2	822885.536	-835477.1824	1	3
3	823554.6489	-836866.6289	1	3
4	823323.7576	-834970.6818	1	3
5	824461.9232	-834304.4253	1	3
6	824438.6521	-835481.2431	1	3

Here, first column is an index(unfortunately is not the same GISJOIN that we have in the others but files are sorted in the same way), 'mx' and 'my' are the coordinates of the center of the census block in meters and corresponds to the values in the next file; ConDis = Number of original congressional -1 (to be pythonic in indexing), Labels = Values of our new clustered districts.

2. x_y_state_n_NAD83_FULL.csv
This is a file that has the original values used to calculate the clustering.

GISJOIN	State	ConDis	mx	my
G01000100211002002	1	2	823716.0154	-835647.7173
G01000100211002002	1	2	822875.7228	-835862.3188
G01000100211002002	1	2	822885.536	-835477.1824
G01000100211002002	1	2	823554.6489	-836866.6289
G01000100211002002	1	2	823323.7576	-834970.6818
G01000100211002002	1	2	824461.9232	-834304.4253
G01000100211002002	1	2	824438.6521	-835481.2431

Here first column is index GISJOIN, this is the right index to get the census block coordinates from the third file and correspond in order (but not value) with the index of file (1). State is the code number for each state (in this case 1 = Alabama); ConDis = Original Congressional District (Without subtracting 1 which means that values in ConDis are not equal at this point); mx and my = correspond to file (1) and are the coordinates of center of census block in meters.

3. [state_name]_block_2010.shp where stat_name is two letter code for state in this case 'AL'
In the shapefile you will find the coordinates and shape for each census block and can be indexed with GISJOIN and correspond to the values in file (2).
