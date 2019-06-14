# Goal
The main purpose of this repository is to provide an easy environment to test different ideas about CPU implementation of **TopK** layers. 

# Methods
Here is the details of implemented algorithms and layers:

| Function Name | Sorting Algorithm |Batch Op| Description |
|:-------------:|:-------------:|:-----:|:-----:|
| SelectionSortTopK | Selection Sort | NO | Function **SelectionSortTopK** uses selection sort algorithm to retrive **K** first elements of descending sorted tensor, stopping algorithm when required number of elements are sorted. | 
| BatchSelectionSortTopK | Selection Sort | YES | Same as **SelectionSortTopK** but takes input tensor of rank three(batchxNxN)| 


# Build
To build the project run these commands at the repository directory:
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

Then executables will be ready to be run.
