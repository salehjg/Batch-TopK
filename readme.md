# Goal
The main purpose of this repository is to provide a working space to test different ideas about CPU implementation of *TopK* layer. 

# Methods
Currently, only selection sort algorithm is tested.

## Selection Sort Based TopK
Function *SelectionSortTopK* uses selection sort algorithm to retrive *K* first elements of descending sorted tensor, stopping algorithm when required number of elements are sorted.

# Build
To build the project run these commands at the repository directory:
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

Then executable will be ready to be run.