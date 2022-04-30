/*
    James William Fletcher (james@voxdsp.com)
        April 2022

    Splits dataset into x & y for keras.

    TRAIN_SIZE needs to be manually updated if you
    generate a new dataset.dat. It is the size in
    bytes of the file divided by 32.

    compile: gcc splitter.c -Ofast -lm -o splitter
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define uint unsigned int

#define TRAIN_SIZE 3341565
float trainset[TRAIN_SIZE*32];

int main(int argc, char** argv)
{
    // load dataset
    FILE* f = fopen("dataset.dat", "rb");
    if(f != NULL)
    {
        if(fread(&trainset[0], 1, TRAIN_SIZE*32, f) != TRAIN_SIZE*32)
        {
            printf("fread() failed.\n");
            exit(0);
        }
        fclose(f);
    }

    // output dataset
    const time_t st = time(0);
    for(size_t i = 0; i < TRAIN_SIZE; i++)
    {
        const uint ofs = i * 8;

        FILE* f = fopen("dataset_x.dat", "ab");
        if(f != NULL)
        {
            size_t r = 0;
            r += fwrite(&trainset[ofs], sizeof(float), 6, f);
            if(r != 6)
            {
                printf("TERMINATED, just wrote corrupted bytes to the dataset! (last %zu bytes).\n", r);
                return 0;
            }
            fclose(f);
        }

        f = fopen("dataset_y.dat", "ab");
        if(f != NULL)
        {
            size_t r = 0;
            r += fwrite(&trainset[ofs+6], sizeof(float), 2, f);
            if(r != 2)
            {
                printf("TERMINATED, just wrote corrupted bytes to the dataset! (last %zu bytes).\n", r);
                return 0;
            }
            fclose(f);
        }
    }

    // done
    printf("Datasets splitted.\nTime Taken: %lu seconds.\n", time(0)-st);
    //getchar();
    return 0;
}