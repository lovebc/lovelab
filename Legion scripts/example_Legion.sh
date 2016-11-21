#!/bin/bash -l
# The above means you're in login shell, i.e. you get all your stuff from .bashrc and .bash_profile
# Check https://wiki.rc.ucl.ac.uk/wiki/Example_Submission_Scripts for more (~up-to-date-ish) general info
# Batch script to run an OpenMP threaded job on Legion with the upgraded
# software stack under SGE.
#
# There are two types of parallelisms here:
# - 'within' node - i.e. within your allocated machine, you can use m cores (threads). It's only helpful to select > 1
#   if your code can take advantage of multiple cores on a single machine. You can request up to 12 threads -
#   - change it in point 3.
# - 'between' node - i.e. you can run N jobs, each with m cores. It's a good solution especially if your code
#   cannot be easily parallelised, but you can split it into N chunks (e.g. analysing each participant's data separately),
#   or if analysing the whole lot would take more than the 4h maximum job runtime. You can also combine 'within' and
#   'between' and end up with N x m threads. Change param 6. for more jobs.
#
# This script consists of two parts:
# 	1. Cluster config - this sets up the environment. See comments or wiki for more.
#		All lines beginning with #$ carry some commands for the queuing system.
#	2. The actual script - this can contain any arbitrary code that will be run for each job.
#		You may need to load / unload some modules, see below. You have access to some
#		variables, e.g. $TMPDIR or $SGE_TASK_ID - see below and https://wiki.duke.edu/display/SCSC/SGE+Env+Vars
#		(note the above is not from UCL, so may not all be available here.)
#		In principle, you can install (user install) any software for which you have a licence,
#		so if you can compile it under Linux/Unix you should be able to run it in a job.
#
# To submit a job to a queue, you simply run:
# qsub jobscript.sh
# from a login node.

####### CLUSTER CONFIG #########
# 1. Force bash as the executing shell.
#$ -S /bin/bash
# 2. Request one hour of wallclock time (format hours:minutes:seconds) PER JOB.
#$ -l h_rt=1:0:0
# 3. Request 12 cores, 2 GPUs (that's an experimenlat option), 1 gigabyte of RAM (!!PER CORE), 15 gigabyte of TMPDIR space
# See here for more about maximum RAM per machine: https://wiki.rc.ucl.ac.uk/wiki/RC_Systems#Legion_technical_specs
#$ -l mem=1G
#$ -pe mpi 12
#$ -l tmpfs=15G
# (optional GPU - RC describe it as 'experimental')
#$ -l gpu=2
# 4. Set the name of the job. This will be visible when you run qstat
#$ -N SciencyMcScienceface
# 5. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/<your_UCL_id>/Scratch/output
# 6. make 100 jobs run with different numbers. You can also request task IDs numbered
# 1 to 1000 with a stride of 10 by doing: -t 1-1000:10
#$ -t 1-100
# 7. This (optional) command transfers all your TMPDIR to your Scratch space.
#
# "Users can automate the transfer of data from $TMPDIR to their scratch space
# by adding the directive #Local2Scratch to their script. At the end of the job,
# files are transferred from $TMPDIR to a directory in scratch with the structure
# <job id>/<job id>.<task id>.<queue>/."
#Local2Scratch

######### ACTUAL SCRIPT ###########
cd $TMPDIR  # This will be automatically saved after the job finishes if you have Local2Scratch
mkdir results
# Module loading / unloading: see here: https://wiki.rc.ucl.ac.uk/wiki/Software
# type `module avail` for a list of all available modules
module unload compilers
module unload mpi
module load r/recommended
export R_input=/home/ucabkop/scripts/SUPERDUPERSCIENCE.R
export R_output=myRjob.out
# $SGE_TASK_ID will have the current job number (see point 6. above)
# Run the R script with this command and some command line params.
# See https://wiki.rc.ucl.ac.uk/wiki/R_on_Legion
R --no-save < $R_input --args $SGE_TASK_ID arg2 arg3 arg4 (...) > $R_output
# say your script saved to results/SciencyMcScienceFaceResults, then you can tar.gz the filder by:
tar -czf results.tgz results/SciencyMcSciencefaceResults
# and remove results
rm -rf results

