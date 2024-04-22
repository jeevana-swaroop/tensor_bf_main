#include <vector>
#include <istream>
#include <fstream>
#include <sstream>
#include <iostream>

#include "params.h"
#include "dataset.h"

void importDataset(
    std::vector< std::vector<INPUT_DATA_TYPE> >* inputVector,
    char* filename)
{
    FILE* fptr;
    fptr = fopen(filename, "r");
    if (NULL == fptr)
    {
        fprintf(stderr, "[Dataset] ~ Could not open the input file %s\n", filename);
        exit(1);
    }
    double check = 0;

    std::vector<INPUT_DATA_TYPE> data;

    int dimCounter = 0;

    while (fscanf(fptr, "%lf, ", &check) == 1 || fscanf(fptr, "%lf ", &check) == 1)
    {
        data.push_back(check);
        dimCounter++;

        if (INPUT_DATA_DIM == dimCounter)
        {
            dimCounter = 0;
            inputVector->push_back(data);
            data.clear();
        }
    }
    fclose(fptr);
}