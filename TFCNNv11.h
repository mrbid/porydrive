/*
--------------------------------------------------
    James William Fletcher (james@voxdsp.com)
        October 2020 - TFCNNv1
        April   2022 - TFCNNv1.1
--------------------------------------------------
    Tiny Fully Connected Neural Network Library
    https://github.com/tfcnn

    - TFCNNv1.1
        Modified for tanh output.
*/

#ifndef TFCNN_H
#define TFCNN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define uint unsigned int

/*
--------------------------------------
    structures
--------------------------------------
*/

// perceptron struct
struct
{
    float* data;
    float* momentum;
    float bias;
    float bias_momentum;
    uint weights;
}
typedef ptron;

// network struct
struct
{
    // hyperparameters
    uint  init;
    uint  activator;
    uint  optimiser;
    uint  batches;
    float rate;
    float gain;
    float dropout;
    float momentum;
    float rmsalpha;

    // layers
    ptron** layer;

    // count
    uint num_inputs;
    uint num_layers;
    uint num_layerunits;

    // backprop
    uint cbatches;
    float** output;
    float foutput;
    float error;
}
typedef network;

/*
--------------------------------------
    ERROR TYPES
--------------------------------------
*/

#define ERROR_UNINITIALISED_NETWORK -1
#define ERROR_TOOFEWINPUTS -2
#define ERROR_TOOFEWLAYERS -3
#define ERROR_TOOSMALL_LAYERSIZE -4
#define ERROR_ALLOC_LAYERS_ARRAY_FAIL -5
#define ERROR_ALLOC_LAYERS_FAIL -6
#define ERROR_ALLOC_OUTPUTLAYER_FAIL -7
#define ERROR_ALLOC_PERCEPTRON_DATAWEIGHTS_FAIL -8
#define ERROR_ALLOC_PERCEPTRON_ALPHAWEIGHTS_FAIL -9
#define ERROR_CREATE_FIRSTLAYER_FAIL -10
#define ERROR_CREATE_HIDDENLAYER_FAIL -11
#define ERROR_CREATE_OUTPUTLAYER_FAIL -12
#define ERROR_ALLOC_OUTPUT_ARRAY_FAIL -13
#define ERROR_ALLOC_OUTPUT_FAIL -14

/*
--------------------------------------
    DEFINES / ENUMS
--------------------------------------
*/

enum 
{
    WEIGHT_INIT_UNIFORM             = 0,
    WEIGHT_INIT_UNIFORM_GLOROT      = 1,
    WEIGHT_INIT_UNIFORM_LECUN       = 2,
    WEIGHT_INIT_UNIFORM_HE          = 3,
    WEIGHT_INIT_NONE                = 4
}
typedef weight_init_type;

enum 
{
    IDENTITY = 0,
    ATAN     = 1,
    TANH     = 2,
    RELU     = 3,
    SIGMOID  = 4
}
typedef activator;

enum 
{
    OPTIM_SGD       = 0,
    OPTIM_MOMENTUM  = 1,
    OPTIM_NESTEROV  = 2,
    OPTIM_ADAGRAD   = 3,
    OPTIM_RMSPROP   = 4
}
typedef optimiser;

/*
--------------------------------------
    random functions
--------------------------------------
*/

#define FAST_PREDICTABLE_MODE

float qRandFloat(const float min, const float max);
float qRandWeight(const float min, const float max);
uint  qRand(const uint min, const uint umax);

/*
--------------------------------------
    accessors
--------------------------------------
*/

void setWeightInit(network* net, const weight_init_type u);
void setOptimiser(network* net, const optimiser u);
void setActivator(network* net, const activator u);
void setBatches(network* net, const uint u);
void setLearningRate(network* net, const float f);
void setGain(network* net, const float f);
void setDropout(network* net, const float f);
void setMomentum(network* net, const float f);
void setRMSAlpha(network* net, const float f);
void randomHyperparameters(network* net);

/*
--------------------------------------
    neural net functions
--------------------------------------
*/

int createNetwork(network* net, const weight_init_type init_type, const uint num_inputs, const uint num_hidden_layers, const uint num_layer_units);
float learnNetwork(network* net, const float* inputs, const float target);
float queryNetwork(network* net, const float* inputs);
void resetNetwork(network* net);
void destroyNetwork(network* net);
int saveWeights(network* net, const char* file);
int loadWeights(network* net, const char* file);

/*
--------------------------------------
    the code ...
--------------------------------------
*/

float qRandFloat(const float min, const float max)
{
    static float rndmax = (float)RAND_MAX;
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
#endif
    return ( ( (((float)rand())+1e-7f) / rndmax ) * (max-min) ) + min;
}

float qRandWeight(const float min, const float max)
{
    static float rndmax = (float)RAND_MAX;
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
#endif
    float pr = 0;
    while(pr == 0) //never return 0
    {
        const float rv2 = ( ( (((float)rand())+1e-7f) / rndmax ) * (max-min) ) + min;
        pr = roundf(rv2 * 100.f) / 100.f; // two decimals of precision
    }
    return pr;
}

uint qRand(const uint min, const uint umax)
{
    static float rndmax = (float)RAND_MAX;
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
#endif
    const uint max = umax + 1;
    return ( ( (((float)rand())+1e-7f) / rndmax ) * (max-min) ) + min;
}

/**********************************************/

static inline float atanDerivative(const float x) //atanf()
{
    return 1.f / (1.f + (x*x));
}

static inline float tanhDerivative(const float x) //tanhf()
{
    return 1.f - (x*x);
}

/**********************************************/

static inline float relu(const float x)
{
    if(x < 0.f){return 0.f;}
    return x;
}

static inline float reluDerivative(const float x)
{
    if(x > 0.f)
        return 1.f;
    else
        return 0.f;
}

/**********************************************/

static inline float sigmoid(const float x)
{
    return 1.f / (1.f + expf(-x));
}

static inline float sigmoidDerivative(const float x)
{
    return x * (1.f - x);
}

/**********************************************/

static inline float Derivative(const float x, const uint type)
{
    if(type == 1)
        return atanDerivative(x);
    else if(type == 2)
        return tanhDerivative(x);
    else if(type == 3)
        return reluDerivative(x);
    else if(type == 4)
        return sigmoidDerivative(x);
    
    return reluDerivative(x); // same as identity derivative
}

static inline float Activator(const float x, const uint type)
{
    if(type == 1)
        return atanf(x);
    else if(type == 2)
        return tanhf(x);
    else if(type == 3)
        return relu(x);
    else if(type == 4)
        return sigmoid(x);

    return x;
}

/**********************************************/

static inline float SGD(network* net, const float input, const float error)
{
    return net->rate * error * input;
}

static inline float Momentum(network* net, const float input, const float error, float* momentum)
{
    // const float err = (_lrate * error * input);
    // const float ret = err + _lmomentum * momentum[0];
    // momentum[0] = err;
    // return ret;

    const float err = (net->rate * error * input) + net->momentum * momentum[0];
    momentum[0] = err;
    return err;
}

static inline float Nesterov(network* net, const float input, const float error, float* momentum)
{
    const float v = net->momentum * momentum[0] + ( net->rate * error * input );
    const float n = v + net->momentum * (v - momentum[0]);
    momentum[0] = v;
    return n;
}

static inline float ADAGrad(network* net, const float input, const float error, float* momentum)
{
    const float err = error * input;
    momentum[0] += err * err;
    return (net->rate / sqrtf(momentum[0] + 1e-7f)) * err;
}

static inline float RMSProp(network* net, const float input, const float error, float* momentum)
{
    const float err = error * input;
    momentum[0] = net->rmsalpha * momentum[0] + (1.f - net->rmsalpha) * (err * err);
    return (net->rate / sqrtf(momentum[0] + 1e-7f)) * err;
}

static inline float Optimiser(network* net, const float input, const float error, float* momentum)
{
    if(net->optimiser == 1)
        return Momentum(net, input, error, momentum);
    else if(net->optimiser == 2)
        return Nesterov(net, input, error, momentum);
    else if(net->optimiser == 3)
        return ADAGrad(net, input, error, momentum);
    else if(net->optimiser == 4)
        return RMSProp(net, input, error, momentum);
    
    return SGD(net, input, error);
}

/**********************************************/

static inline float doPerceptron(const float* in, ptron* p)
{
    float ro = 0.f;
    for(uint i = 0; i < p->weights; i++)
        ro += in[i] * p->data[i];
    ro += p->bias;

    return ro;
}

/**********************************************/

int createPerceptron(ptron* p, const uint weights, const float d)
{
    p->data = malloc(weights * sizeof(float));
    if(p->data == NULL)
        return ERROR_ALLOC_PERCEPTRON_DATAWEIGHTS_FAIL;

    p->momentum = malloc(weights * sizeof(float));
    if(p->momentum == NULL)
    {
        free(p->data);
        return ERROR_ALLOC_PERCEPTRON_ALPHAWEIGHTS_FAIL;
    }

    p->weights = weights;

    for(uint i = 0; i < p->weights; i++)
    {
        p->data[i] = qRandWeight(-d, d);
        p->momentum[i] = 0.f;
    }

    p->bias = 0.f;
    p->bias_momentum = 0.f;

    return 0;
}

void resetPerceptron(ptron* p, const float d)
{
    for(uint i = 0; i < p->weights; i++)
    {
        p->data[i] = qRandWeight(-d, d);
        p->momentum[i] = 0.f;
    }

    p->bias = 0.f;
    p->bias_momentum = 0.f;
}

void setWeightInit(network* net, const weight_init_type u)
{
    if(net == NULL){return;}
    net->init = u;
}

void setOptimiser(network* net, const optimiser u)
{
    if(net == NULL){return;}
    net->optimiser = u;
}

void setActivator(network* net, const activator u)
{
    if(net == NULL){return;}
    net->activator = u;
}

void setBatches(network* net, const uint u)
{
    if(net == NULL){return;}
    if(u == 0)
        net->batches = 1;
    else
        net->batches = u;
}

void setLearningRate(network* net, const float f)
{
    if(net == NULL){return;}
    net->rate = f;
}

void setGain(network* net, const float f)
{
    if(net == NULL){return;}
    net->gain = f;
}

void setDropout(network* net, const float f)
{
    if(net == NULL){return;}
    net->dropout = f;
}

void setMomentum(network* net, const float f)
{
    if(net == NULL){return;}
    net->momentum = f;
}

void setRMSAlpha(network* net, const float f)
{
    if(net == NULL){return;}
    net->rmsalpha = f;
}

void randomHyperparameters(network* net)
{
    if(net == NULL){return;}
        
    net->init       = qRand(0, 3);
    net->activator  = qRand(1, 4);
    net->optimiser  = qRand(0, 4);
    net->rate       = qRandFloat(0.001f, 0.03f);
    net->dropout    = qRandFloat(0.f, 0.3f);
    net->momentum   = qRandFloat(0.1f, 0.9f);
    net->rmsalpha   = qRandFloat(0.2f, 0.99f);
}

int createNetwork(network* net, const uint init_weights_type, const uint inputs, const uint hidden_layers, const uint layers_size)
{
    const uint layers = hidden_layers+2;

    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(inputs < 1)
        return ERROR_TOOFEWINPUTS;
    if(layers < 3)
        return ERROR_TOOFEWLAYERS;
    if(layers_size < 1)
        return ERROR_TOOSMALL_LAYERSIZE;

    // init net hyper parameters to some default
    net->num_layerunits = layers_size;
    net->num_inputs = inputs;
    net->num_layers = layers;
    net->init       = init_weights_type;
    net->activator  = 2;
    net->optimiser  = 1;
    net->batches    = 3;
    net->rate       = 0.01f;
    net->gain       = 1.0f;
    net->dropout    = 0.3f;
    net->momentum   = 0.1f;
    net->rmsalpha   = 0.2f;
    net->cbatches   = 0;
    net->error      = 0.f;
    net->foutput    = 0.f;
    
    // create layers
    net->output = malloc((layers-1) * sizeof(float*));
    if(net->output == NULL)
    {
        destroyNetwork(net);
        return ERROR_ALLOC_OUTPUT_ARRAY_FAIL;
    }
    for(int i = 0; i < layers-1; i++)
    {
        net->output[i] = malloc(layers_size * sizeof(float));
        if(net->output[i] == NULL)
        {
            destroyNetwork(net);
            return ERROR_ALLOC_OUTPUT_FAIL;
        }
    }

    net->layer = malloc(layers * sizeof(ptron*));
    if(net->layer == NULL)
    {
        destroyNetwork(net);
        return ERROR_ALLOC_LAYERS_ARRAY_FAIL;
    }
    for(int i = 0; i < layers-1; i++)
    {
        net->layer[i] = malloc(layers_size * sizeof(ptron));
        if(net->layer[i] == NULL)
        {
            destroyNetwork(net);
            return ERROR_ALLOC_LAYERS_FAIL;
        }
    }

    net->layer[layers-1] = malloc(sizeof(ptron));
    if(net->layer[layers-1] == NULL)
    {
        destroyNetwork(net);
        return ERROR_ALLOC_OUTPUTLAYER_FAIL;
    }

    // init weight
    float d = 1.f; //WEIGHT_INIT_UNIFORM / WEIGHT_INIT_NORMAL
    if(init_weights_type == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtf(6.0f/(inputs+layers_size));
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrtf(3.0f/inputs);
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_HE)
        d = sqrtf(6.0f/inputs);

    // create first layer perceptrons
    for(int i = 0; i < layers_size; i++)
    {
        if(createPerceptron(&net->layer[0][i], inputs, d) < 0)
        {
            destroyNetwork(net);
            return ERROR_CREATE_FIRSTLAYER_FAIL;
        }
    }
    
    // weight init
    if(init_weights_type == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtf(6.0f/(layers_size+layers_size));
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrtf(3.0f/layers_size);
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_HE)
        d = sqrtf(6.0f/layers_size);

    // create hidden layers
    for(uint i = 1; i < layers-1; i++)
    {
        for(int j = 0; j < layers_size; j++)
        {
            if(createPerceptron(&net->layer[i][j], layers_size, d) < 0)
            {
                destroyNetwork(net);
                return ERROR_CREATE_HIDDENLAYER_FAIL;
            }
        }   
    }

    // weight init
    if(init_weights_type == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtf(6.0f/(layers_size+1));

    // create output layer
    if(createPerceptron(&net->layer[layers-1][0], layers_size, d) < 0)
    {
        destroyNetwork(net);
        return ERROR_CREATE_OUTPUTLAYER_FAIL;
    }

    // done
    return 0;
}

float queryNetwork(network* net, const float* inputs)
{
    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(net->layer == NULL)
        return ERROR_UNINITIALISED_NETWORK;

    // outputs per layer / unit
    float of[net->num_layers-1][net->num_layerunits];

    // input layer
    for(int i = 0; i < net->num_layerunits; i++)
        of[0][i] = Activator(doPerceptron(inputs, &net->layer[0][i]), net->activator);

    // hidden layers
    for(int i = 1; i < net->num_layers-1; i++)
        for(int j = 0; j < net->num_layerunits; j++)
            of[i][j] = Activator(doPerceptron(&of[i-1][0], &net->layer[i][j]), net->activator);

    // binary classifier output layer
    const float output = tanh(doPerceptron(&of[net->num_layers-2][0], &net->layer[net->num_layers-1][0]));

    // return output
    return output;
}

float trainNetwork(network* net, const float* inputs, const float target)
{
    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(net->layer == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    
/**************************************
    Forward Prop
**************************************/

    // outputs per layer / unit
    float of[net->num_layers-1][net->num_layerunits];

    // input layer
    for(int i = 0; i < net->num_layerunits; i++)
        of[0][i] = Activator(doPerceptron(inputs, &net->layer[0][i]), net->activator);

    // hidden layers
    for(int i = 1; i < net->num_layers-1; i++)
        for(int j = 0; j < net->num_layerunits; j++)
            of[i][j] = Activator(doPerceptron(&of[i-1][0], &net->layer[i][j]), net->activator);

    // binary classifier output layer
    const float output = tanh(doPerceptron(&of[net->num_layers-2][0], &net->layer[net->num_layers-1][0]));

/**************************************
    Backward Prop Error
**************************************/

    // reset accumulators if cbatches has been reset
    if(net->cbatches == 0)
    {
        for(int i = 0; i < net->num_layers-1; i++)
            memset(net->output[i], 0x00, net->num_layerunits * sizeof(float));

        net->foutput = 0.f;
        net->error = 0.f;
    }

    // batch accumulation of outputs
    net->foutput += output;
    for(int i = 0; i < net->num_layers-1; i++)
        for(int j = 0; j < net->num_layerunits; j++)
            net->output[i][j] += of[i][j];

    // accumulate output error
    net->error += target - output;

    // batching controller
    net->cbatches++;
    if(net->cbatches < net->batches)
    {
        return output;
    }
    else
    {
        // divide accumulators to mean
        net->error /= net->batches;
        net->foutput /= net->batches;

        for(int i = 0; i < net->num_layers-1; i++)
            for(int j = 0; j < net->num_layerunits; j++)
                net->output[i][j] /= net->batches;

        // reset batcher
        net->cbatches = 0;
    }

    // early return if error is 0
    if(net->error == 0.f)
        return output;

    // define error buffers
    float ef[net->num_layers-1][net->num_layerunits];

    // output (binary classifier) derivative error
    const float eout = net->gain * tanhDerivative(net->foutput) * net->error;

    // output 'derivative error layer' of layer before/behind the output layer
    float ler = 0.f;
    for(int j = 0; j < net->layer[net->num_layers-1][0].weights; j++)
        ler += net->layer[net->num_layers-1][0].data[j] * eout;
    ler += net->layer[net->num_layers-1][0].bias * eout;
    for(int i = 0; i < net->num_layerunits; i++)
        ef[net->num_layers-2][i] = net->gain * Derivative(net->output[net->num_layers-2][i], net->activator) * ler;

    // output derivative error of all other layers
    for(int i = net->num_layers-3; i >= 0; i--)
    {
        // compute total error of layer above w.r.t all weights and units of the above layer
        float ler = 0.f;
        for(int j = 0; j < net->num_layerunits; j++)
        {
            for(int k = 0; k < net->layer[i+1][j].weights; k++)
                ler += net->layer[i+1][j].data[k] * ef[i+1][j];
            ler += net->layer[i+1][j].bias * ef[i+1][j];
        }
        // propagate that error to into the error variable of each unit of the current layer
        for(int j = 0; j < net->num_layerunits; j++)
            ef[i][j] = net->gain * Derivative(net->output[i][j], net->activator) * ler;
    }

/**************************************
    Update Weights
**************************************/
    
    // update input layer weights
    for(int j = 0; j < net->num_layerunits; j++)
    {
        if(net->dropout != 0 && qRandFloat(0, 1) <= net->dropout)
            continue;

        for(int k = 0; k < net->layer[0][j].weights; k++)
            net->layer[0][j].data[k] += Optimiser(net, inputs[k], ef[0][j], &net->layer[0][j].momentum[k]);

        net->layer[0][j].bias += Optimiser(net, 1, ef[0][j], &net->layer[0][j].bias_momentum);
    }

    // update hidden layer weights
    for(int i = 1; i < net->num_layers-1; i++)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            if(net->dropout != 0 && qRandFloat(0, 1) <= net->dropout)
                continue;

            for(int k = 0; k < net->layer[i][j].weights; k++)
                net->layer[i][j].data[k] += Optimiser(net, net->output[i-1][j], ef[i][j], &net->layer[i][j].momentum[k]);

            net->layer[i][j].bias += Optimiser(net, 1, ef[i][j], &net->layer[i][j].bias_momentum);
        }
    }

    // update output layer weights
    for(int j = 0; j < net->layer[net->num_layers-1][0].weights; j++)
        net->layer[net->num_layers-1][0].data[j] += Optimiser(net, net->output[net->num_layers-2][j], eout, &net->layer[net->num_layers-1][0].momentum[j]);

    net->layer[net->num_layers-1][0].bias += Optimiser(net, 1, eout, &net->layer[net->num_layers-1][0].bias_momentum);

    // done, return forward prop output
    return output;
}

void resetNetwork(network* net)
{
    // validate
    if(net == NULL)
        return;
    if(net->layer == NULL)
        return;

    // reset batching counter
    for(int i = 0; i < net->num_layers-1; i++)
        memset(net->output[i], 0x00, net->num_layerunits * sizeof(float));
    net->cbatches = 0;
    net->foutput = 0.f;
    net->error = 0.f;
    
    // init weight
    float d = 1.f; //WEIGHT_INIT_RANDOM
    if(net->init == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtf(6.0f/(net->num_inputs+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrtf(3.0f/net->num_inputs);
    else if(net->init == WEIGHT_INIT_UNIFORM_HE)
        d = sqrtf(6.0f/net->num_inputs);

    // reset first layer perceptrons
    for(int i = 0; i < net->num_layerunits; i++)
        resetPerceptron(&net->layer[0][i], d);
    
    // weight init
    if(net->init == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtf(6.0f/(net->num_layerunits+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrtf(3.0f/net->num_layerunits);
    else if(net->init == WEIGHT_INIT_UNIFORM_HE)
        d = sqrtf(6.0f/net->num_layerunits);

    // reset hidden layers
    for(uint i = 1; i < net->num_layers-1; i++)
        for(int j = 0; j < net->num_layerunits; j++)
            resetPerceptron(&net->layer[i][j], d);

    // weight init
    if(net->init == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtf(6.0f/(net->num_layerunits+1));

    // reset output layer
    resetPerceptron(&net->layer[net->num_layers-1][0], d);
}

void destroyNetwork(network* net)
{
    // validate
    if(net == NULL)
        return;
    if(net->layer == NULL)
        return;

    // free all perceptron data, percepron units and layers
    for(int i = 0; i < net->num_layers-1; i++)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            free(net->layer[i][j].data);
            free(net->layer[i][j].momentum);
        }
        free(net->layer[i]);
    }
    free(net->layer[net->num_layers-1][0].data);
    free(net->layer[net->num_layers-1][0].momentum);
    free(net->layer[net->num_layers-1]);
    free(net->layer);

    // free output buffers
    for(int i = 0; i < net->num_layers-1; i++)
        free(net->output[i]);
    free(net->output);
}

int saveWeights(network* net, const char* file)
{
    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(net->layer == NULL)
        return ERROR_UNINITIALISED_NETWORK;

    FILE* f = fopen(file, "wb");
    if(f != NULL)
    {
        for(int i = 0; i < net->num_layers-1; i++)
        {
            for(int j = 0; j < net->num_layerunits; j++)
            {
                if(fwrite(&net->layer[i][j].data[0], 1, net->layer[i][j].weights*sizeof(float), f) != net->layer[i][j].weights*sizeof(float))
                {
                    fclose(f);
                    return -1;
                }
                
                if(fwrite(&net->layer[i][j].momentum[0], 1, net->layer[i][j].weights*sizeof(float), f) != net->layer[i][j].weights*sizeof(float))
                {
                    fclose(f);
                    return -1;
                }

                if(fwrite(&net->layer[i][j].bias, 1, sizeof(float), f) != sizeof(float))
                {
                    fclose(f);
                    return -1;
                }
                
                if(fwrite(&net->layer[i][j].bias_momentum, 1, sizeof(float), f) != sizeof(float))
                {
                    fclose(f);
                    return -1;
                }
            }
        }

        if(fwrite(&net->layer[net->num_layers-1][0].data[0], 1, net->layer[net->num_layers-1][0].weights*sizeof(float), f) != net->layer[net->num_layers-1][0].weights*sizeof(float))
        {
            fclose(f);
            return -1;
        }
        
        if(fwrite(&net->layer[net->num_layers-1][0].momentum[0], 1, net->layer[net->num_layers-1][0].weights*sizeof(float), f) != net->layer[net->num_layers-1][0].weights*sizeof(float))
        {
            fclose(f);
            return -1;
        }

        if(fwrite(&net->layer[net->num_layers-1][0].bias, 1, sizeof(float), f) != sizeof(float))
        {
            fclose(f);
            return -1;
        }
        
        if(fwrite(&net->layer[net->num_layers-1][0].bias_momentum, 1, sizeof(float), f) != sizeof(float))
        {
            fclose(f);
            return -1;
        }

        fclose(f);
    }

    return 0;
}

int loadWeights(network* net, const char* file)
{
    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(net->layer == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    
    FILE* f = fopen(file, "rb");
    if(f == NULL)
        return -1;

    for(int i = 0; i < net->num_layers-1; i++)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            if(fread(&net->layer[i][j].data[0], 1, net->layer[i][j].weights*sizeof(float), f) != net->layer[i][j].weights*sizeof(float))
            {
                fclose(f);
                return -1;
            }

            if(fread(&net->layer[i][j].momentum[0], 1, net->layer[i][j].weights*sizeof(float), f) != net->layer[i][j].weights*sizeof(float))
            {
                fclose(f);
                return -1;
            }

            if(fread(&net->layer[i][j].bias, 1, sizeof(float), f) != sizeof(float))
            {
                fclose(f);
                return -1;
            }

            if(fread(&net->layer[i][j].bias_momentum, 1, sizeof(float), f) != sizeof(float))
            {
                fclose(f);
                return -1;
            }
        }
    }

    if(fread(&net->layer[net->num_layers-1][0].data[0], 1, net->layer[net->num_layers-1][0].weights*sizeof(float), f) != net->layer[net->num_layers-1][0].weights*sizeof(float))
    {
        fclose(f);
        return -1;
    }
    
    if(fread(&net->layer[net->num_layers-1][0].momentum[0], 1, net->layer[net->num_layers-1][0].weights*sizeof(float), f) != net->layer[net->num_layers-1][0].weights*sizeof(float))
    {
        fclose(f);
        return -1;
    }

    if(fread(&net->layer[net->num_layers-1][0].bias, 1, sizeof(float), f) != sizeof(float))
    {
        fclose(f);
        return -1;
    }

    if(fread(&net->layer[net->num_layers-1][0].bias_momentum, 1, sizeof(float), f) != sizeof(float))
    {
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

#endif

