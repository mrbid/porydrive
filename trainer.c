/*
    James William Fletcher (github.com/mrbid)
        April 2022

    Trains the neural network

    TRAIN_SIZE needs to be manually updated if you
    generate a new dataset.dat. It is the size in
    bytes of the file divided by 32.

    compile: gcc trainer.c -Ofast -lm -o train
    (compile with clang might yeild faster performance)
*/

#include <time.h>
#include <unistd.h> // nice
#include <errno.h>

#include "TFCNNv11.h"
#include "globaldef.h"

network net;

#define TRAIN_SIZE 3341565
float trainset[TRAIN_SIZE*32];

void trainSteeringAgent()
{
    // reset network
    resetNetwork(&net);

    // train network
    for(uint i = 0; i < TRAIN_SIZE; i++)
    {
        const uint ofs = i * 8;
        const float target = trainset[ofs+6]; // steering
        trainNetwork(&net, &trainset[ofs], target);
    }

    // save weights
    saveWeights(&net, "steeragent_weights.dat");
}

void trainGasingAgent()
{
    // reset network
    resetNetwork(&net);
    
    // train network
    for(uint i = 0; i < TRAIN_SIZE; i++)
    {
        const uint ofs = i * 8;
        const float target = trainset[ofs+7]; // gassing
        trainNetwork(&net, &trainset[ofs], target);
    }

    // save weights
    saveWeights(&net, "gasagent_weights.dat");
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        printf("Specify net 0 or 1 to train as first argument.\n");
        return 0;
    }

    // max performance
    errno = 0;
    if(nice(-20) < 0)
    {
        while(errno != 0)
        {
            errno = 0;
            if(nice(-20) < 0)
                printf("Attempting to set process to nice of -20 (run with sudo)...\n");
            sleep(1);
        }
    }

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

    // create network
    const int ret = createNetwork(&net, WEIGHT_INIT_UNIFORM_LECUN, 6, HIDDEN_LAYERS, HIDDEN_SIZE);
    if(ret < 0)
    {
        printf("createNetwork() failed: %i\n", ret);
        exit(0);
    }

    // set hyper parameters
    setOptimiser(&net, OPTIM_ADAGRAD);
    setDropout(&net, 0.3f);
    setBatches(&net, 32);
    setWeightInit(&net, WEIGHT_INIT_UNIFORM_LECUN);
    setActivator(&net, TANH);

    // train network
    const time_t st = time(0);
    if(argv[1][0] == '0')
    {
        printf("Training Steering Agent.\n");
        trainSteeringAgent();
    }
    else
    {
        printf("Training Gasing Agent.\n");
        trainGasingAgent();
    }

    // done
    printf("Datasets generated.\nTime Taken: %lu seconds.\n", time(0)-st);
    //getchar();
    return 0;
}
