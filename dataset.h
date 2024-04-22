#ifndef DATASET_H
#define DATASET_H

#include <vector>

#include "params.h"

void importDataset(
    std::vector< std::vector<INPUT_DATA_TYPE> >* inputVector,
    char* filename);

#endif