/*
--------------------------------------------------
    James William Fletcher (james@voxdsp.com)
        November 2020 - TFCNNv2.1
        April    2022 - TFCNNv2.1.1
--------------------------------------------------
    Tiny Fully Connected Neural Network Library
    https://github.com/tfcnn

    - TFCNNv2.1.1
        Modified for tanh output.

    This version uses derivatives w.r.t x rather
    than w.r.t f(x) which is what TFCNNv2 does.
*/

#ifndef TFCNN_H
#define TFCNN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <sys/file.h>

#include <stdint.h>
#include <unistd.h>

#ifndef NOSSE
    #include <x86intrin.h>
#endif

#include "vec.h"
#define qrandf randf

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
    float wdropout;
    float momentum;
    float rmsalpha;
    float epsilon;

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

    // selu alpha dropout
    float drop_a;
    float drop_b;
    float drop_wa;
    float drop_wb;
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
    WEIGHT_INIT_UNIFORM_LECUN_POW   = 3,
    WEIGHT_INIT_NORMAL              = 4,
    WEIGHT_INIT_NORMAL_GLOROT       = 5,
    WEIGHT_INIT_NORMAL_LECUN        = 6,
    WEIGHT_INIT_NORMAL_LECUN_POW    = 7
}
typedef weight_init_type;

enum
{
    SELU        = 0,
    GELU        = 1,
    MISH        = 2,
    ISRU        = 3,
    SQNL        = 4,
    SINC        = 5
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

// quick randoms
float qRandNormal();
float qRandFloat(const float min, const float max);
float qRandWeight(const float min, const float max);
uint  qRand(const uint min, const uint umax);

// slower randoms with higher entropy [make sure FAST_PREDICTABLE_MODE is undefined]
float uRandNormal();
float uRandFloat(const float min, const float max);
float uRandWeight(const float min, const float max);
uint  uRand(const uint min, const uint umax);

// seed with high granularity
void newSRAND();

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
void setUnitDropout(network* net, const float f); //Dropout
void setWeightDropout(network* net, const float f); //Drop Connect
void setMomentum(network* net, const float f); //SGDM & NAG
void setRMSAlpha(network* net, const float f);
void setEpsilon(network* net, const float f); //ADA & RMS
void randomHyperparameters(network* net);

/*
--------------------------------------
    neural net functions
--------------------------------------
*/

int createNetwork(network* net, const weight_init_type init_type, const uint num_inputs, const uint num_hidden_layers, const uint num_layer_units, const uint default_settings);
float trainNetwork(network* net, const float* inputs, const float target);
float queryNetwork(network* net, const float* inputs);
void resetNetwork(network* net);
void destroyNetwork(network* net);
int saveNetwork(network* net, const char* file);
int loadNetwork(network* net, const char* file);

/*
--------------------------------------
    the code ...
--------------------------------------
*/

float urandf()
{
    static const float FLOAT_UINT64_MAX = (float)UINT64_MAX;
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint64_t s = 0;
    ssize_t result = read(f, &s, sizeof(uint64_t));
    close(f);
    return (((float)s)+1e-7f) / FLOAT_UINT64_MAX;
}

float qRandFloat(const float min, const float max)
{
    static const float rmax = (float)RAND_MAX;
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srandf(time(0));
        ls = time(0) + 33;
    }
#endif
    return ( qrandf() * (max-min) ) + min;
}

float uRandFloat(const float min, const float max)
{
    static const float rmax = (float)RAND_MAX;
#ifdef FAST_PREDICTABLE_MODE
    return qRandFloat(min, max);
#else
    return ( urandf() * (max-min) ) + min;
#endif
}

float qRandWeight(const float min, const float max)
{
    static const float rmax = (float)RAND_MAX;
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srandf(time(0));
        ls = time(0) + 33;
    }
#endif
    float pr = 0;
    while(pr == 0) //never return 0
    {
        const float rv2 = ( qrandf() * (max-min) ) + min;
        pr = roundf(rv2 * 100) / 100; // two decimals of precision
    }
    return pr;
}

float uRandWeight(const float min, const float max)
{
    static const float rmax = (float)RAND_MAX;
#ifdef FAST_PREDICTABLE_MODE
    return qRandWeight(min, max);
#else
    float pr = 0;
    while(pr == 0) //never return 0
    {
        const float rv2 = ( urandf() * (max-min) ) + min;
        pr = roundf(rv2 * 100) / 100; // two decimals of precision
    }
    return pr;
#endif
}

uint qRand(const uint min, const uint umax)
{
    static const float rmax = (float)RAND_MAX;
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srandf(time(0));
        ls = time(0) + 33;
    }
#endif
    const uint max = umax + 1;
    return ( qrandf() * (max-min) ) + min;
}

uint uRand(const uint min, const uint umax)
{
    static const float rmax = (float)RAND_MAX;
#ifdef FAST_PREDICTABLE_MODE
    return qRand(min, umax);
#else
    const uint max = umax + 1;
    return ( urandf() * (max-min) ) + min;
#endif
}

float qRandNormal() // Box Muller
{
    static const float rmax = (float)RAND_MAX;
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srandf(time(0));
        ls = time(0) + 33;
    }
#endif
    float u = qrandf() * 2.f - 1.f;
    float v = qrandf() * 2.f - 1.f;
    float r = u * u + v * v;
    while(r == 0.f || r > 1.f)
    {
        u = qrandf() * 2.f - 1.f;
        v = qrandf() * 2.f - 1.f;
        r = u * u + v * v;
    }
    return u * sqrtps(-2.f * logf(r) / r);
}

float uRandNormal()
{
    static const float rmax = (float)RAND_MAX;
#ifdef FAST_PREDICTABLE_MODE
    return qRandNormal();
#else
    float u = urandf() * 2.f - 1.f;
    float v = urandf() * 2.f - 1.f;
    float r = u * u + v * v;
    while(r == 0.f || r > 1.f)
    {
        u = urandf() * 2.f - 1.f;
        v = urandf() * 2.f - 1.f;
        r = u * u + v * v;
    }
    return u * sqrtps(-2.f * logf(r) / r);
#endif
}

void newSRAND()
{
    struct timespec c;
    clock_gettime(CLOCK_MONOTONIC, &c);
    srandf(time(0)+c.tv_nsec);
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

static inline float tanhDerivative(const float x) //tanhf()
{
    return 1.f - (x*x);
}

/**********************************************/

static inline float selu(const float x)
{
    //if(x < 0.f){return 1.0507f * (1.67326f * expf(x) - 1.67326f);}
    if(x < 0.f){return 1.0507f * 1.67326f * (expf(x) - 1.f);} // correct
    return 1.0507f * x;
}

static inline float seluDerivative(const float x)
{
    if(x < 0.f){return 1.0507f * (1.67326f * expf(x));}
    return 1.0507f;
}

/**********************************************/

static inline float sqnl(const float x)
{
    if(x > 2.0f){return 1.f;}
    else if(x < -2.0f){return -1.0f;}
    else if(x <= 2.0f && x >= 0.f){return (x-(x*x))*0.25f;}
    return (x+(x*x))*0.25f;
}

static inline float sqnlDerivative(const float x)
{
    if(x > 0.f){return 1.f - x * 0.5f;}
    return 1.f + x * 0.5f;
}

/**********************************************/

static inline float gelu(const float x)
{
    return 0.5f * x * (1.f + tanhf( 0.45015815796f * ( x + 0.044715f * (x*x*x) ) ));
}

static inline float geluDerivative(const float x)
{
    const double x3 = x*x*x;
    const double s2 = 1.f / coshf(0.0356774f*x3 + 0.797885f*x);
    return 0.5f * tanhf(0.0356774f*x3 + 0.797885f*x) + (0.0535161f*x3 + 0.398942f*x) * (s2 * s2) + 0.5f;
}

/**********************************************/

static inline float isru(const float x)
{
    return x / sqrtps(1.f+x*x);
}

static inline float isruDerivative(const float x)
{
    const float x1 = rsqrtss(1.f + x*x);
    return x1*x1*x1;
}

/**********************************************/

static inline float sinc(const float x)
{
    if(x == 0.f){return 1.f;}
    return sinf(x) / x;
}

static inline float sincDerivative(const float x)
{
    if(x == 0.f){return 0.f;}
    return (cosf(x) / x) - (sinf(x) / (x*x));
}

/**********************************************/

static inline float mish(const float x)
{
    return x * tanhf(logf(expf(x) + 1.f));
}

static inline float mishDerivative(const float x)
{
    const float sech = 1.f / coshf(logf(expf(x) + 1.f));
    return tanhf(logf(expf(x) + 1.f)) + (  (x * expf(x) * (sech*sech)) / (expf(x) + 1.f)  );
}

/**********************************************/
static inline float Derivative(const float x, const network* net)
{
    if(net->activator == 1)
        return geluDerivative(x);
    else if(net->activator == 2)
        return mishDerivative(x);
    else if(net->activator == 3)
        return isruDerivative(x);
    else if(net->activator == 4)
        return sqnlDerivative(x);
    else if(net->activator == 5)
        return sincDerivative(x);
    
    return seluDerivative(x);
}

static inline float Activator(const float x, const network* net)
{
    if(net->activator == 1)
        return gelu(x);
    else if(net->activator == 2)
        return mish(x);
    else if(net->activator == 3)
        return isru(x);
    else if(net->activator == 4)
        return sqnl(x);
    else if(net->activator == 5)
        return sinc(x);

    return selu(x);
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
    return (net->rate / sqrtps(momentum[0] + net->epsilon)) * err;
}

static inline float RMSProp(network* net, const float input, const float error, float* momentum)
{
    const float err = error * input;
    momentum[0] = net->rmsalpha * momentum[0] + (1 - net->rmsalpha) * (err * err);
    return (net->rate / sqrtps(momentum[0] + net->epsilon)) * err;
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

static inline float doDropout(const network* net, const float f, const uint type)
{
    if(type == 1) // SELU dropout
    {
        return f * net->drop_a + net->drop_b;
    }
    else if(type == 2) // weight dropout
    {
        if(uRandFloat(0, 1) <= net->wdropout)
        {
            if(net->activator == SELU) // SELU dropout is not meant to be applied on a weight dropout basis
                return f * net->drop_wa + net->drop_wb;
            else
                return 0.f;
        }
    }
    return f;
}

/**********************************************/

int createPerceptron(ptron* p, const uint weights, const float d, const weight_init_type wit)
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
        if(wit < 4)
            p->data[i] = qRandWeight(-d, d); // uniform
        else
            p->data[i] = qRandNormal() * d;  // normal

        p->momentum[i] = 0.f;
    }

    p->bias = 0.f;
    p->bias_momentum = 0.f;

    return 0;
}

void resetPerceptron(ptron* p, const float d, const weight_init_type wit)
{
    for(uint i = 0; i < p->weights; i++)
    {
        if(wit < 4)
            p->data[i] = qRandWeight(-d, d); // uniform
        else
            p->data[i] = qRandNormal() * d;  // normal
        
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

void setUnitDropout(network* net, const float f)
{
    if(net == NULL){return;}
    net->dropout = f;

    const float d1 = 1.f - net->dropout;
    net->drop_a = powf(net->dropout + 3.090895504f * net->dropout * d1, -0.5f);
    net->drop_b = -net->drop_a * (d1 * -1.758094282f);
}

void setWeightDropout(network* net, const float f) // THIS IS NON STANDARD / ILLEGAL
{
    if(net == NULL){return;}
    net->wdropout = f;

    const float d1 = 1.f - net->wdropout;
    net->drop_wa = powf(net->wdropout + 3.090895504f * net->wdropout * d1, -0.5f);
    net->drop_wb = -net->drop_wa * (d1 * -1.758094282f);
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

void setEpsilon(network* net, const float f)
{
    if(net == NULL){return;}
    net->epsilon = f;
}

void randomHyperparameters(network* net)
{
    if(net == NULL){return;}
        
    net->init       = uRand(0, 7);
    net->activator  = uRand(0, 5);
    net->optimiser  = uRand(0, 4);
    net->rate       = uRandFloat(0.001f, 0.1f);
    net->dropout    = uRandFloat(0.f, 0.99f);
    net->wdropout   = uRandFloat(0.f, 0.99f);
    net->momentum   = uRandFloat(0.01f, 0.99f);
    net->rmsalpha   = uRandFloat(0.01f, 0.99f);
    net->epsilon    = uRandFloat(1e-8f, 1e-5f);
}

int createNetwork(network* net, const uint init_weights_type, const uint inputs, const uint hidden_layers, const uint layers_size, const uint default_settings)
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
    if(default_settings == 1)
    {
        net->activator  = 0;
        net->optimiser  = 2;
        net->batches    = 3;
        net->rate       = 0.01f;
        net->gain       = 1.0f;
        net->dropout    = 0.3f;
        net->wdropout   = 0.f;
        net->momentum   = 0.1f;
        net->rmsalpha   = 0.2f;
        net->epsilon    = 1e-7f;
    }
    net->cbatches   = 0;
    net->error      = 0.f;
    net->foutput    = 0.f;

    float d1 = 1.f - net->dropout;
    net->drop_a = powf(net->dropout + 3.090895504f * net->dropout * d1, -0.5f);
    net->drop_b = -net->drop_a * (d1 * -1.758094282f);
    
    d1 = 1.f - net->wdropout;
    net->drop_wa = powf(net->wdropout + 3.090895504f * net->wdropout * d1, -0.5f);
    net->drop_wb = -net->drop_wa * (d1 * -1.758094282f);
    
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
    float d = 1; //WEIGHT_INIT_UNIFORM / WEIGHT_INIT_NORMAL
    if(init_weights_type == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtps(6.0f/(inputs+layers_size));
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrtps(3.0f/inputs);
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_LECUN_POW)
        d = powf(inputs, 0.5f);
    else if(init_weights_type == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrtps(2.0f/(inputs+layers_size));
    else if(init_weights_type == WEIGHT_INIT_NORMAL_LECUN)
        d = sqrtps(1.0f/inputs);

    // create first layer perceptrons
    for(int i = 0; i < layers_size; i++)
    {
        if(createPerceptron(&net->layer[0][i], inputs, d, net->init) < 0)
        {
            destroyNetwork(net);
            return ERROR_CREATE_FIRSTLAYER_FAIL;
        }
    }
    
    // weight init
    if(init_weights_type == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtps(6.0f/(layers_size+layers_size));
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrtps(3.0f/layers_size);
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_LECUN_POW || init_weights_type == WEIGHT_INIT_NORMAL_LECUN_POW)
        d = powf(layers_size, 0.5f);
    else if(init_weights_type == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrtps(2.0f/(layers_size+layers_size));
    else if(init_weights_type == WEIGHT_INIT_NORMAL_LECUN)
        d = sqrtps(1.0f/layers_size);

    // create hidden layers
    for(uint i = 1; i < layers-1; i++)
    {
        for(int j = 0; j < layers_size; j++)
        {
            if(createPerceptron(&net->layer[i][j], layers_size, d, net->init) < 0)
            {
                destroyNetwork(net);
                return ERROR_CREATE_HIDDENLAYER_FAIL;
            }
        }
    }

    // weight init
    if(init_weights_type == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtps(6.0f/(layers_size+1));
    else if(init_weights_type == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrtps(2.0f/(layers_size+1));

    // create output layer
    if(createPerceptron(&net->layer[layers-1][0], layers_size, d, net->init) < 0)
    {
        destroyNetwork(net);
        return ERROR_CREATE_OUTPUTLAYER_FAIL;
    }

    // done
    return 0;
}

float queryNetwork(network* net, const float* inputs)
{
    // validate [it's ok, the output should be sigmoid 0-1 otherwise]
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(net->layer == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    
/**************************************
    Forward Prop
**************************************/

    // outputs per layer / unit
    float of[net->num_layers-1][net->num_layerunits];
    float output = 0;

    // input layer
    for(int i = 0; i < net->num_layerunits; i++)
        of[0][i] = Activator(doPerceptron(inputs, &net->layer[0][i]), net);

    // hidden layers
    for(int i = 1; i < net->num_layers-1; i++)
        for(int j = 0; j < net->num_layerunits; j++)
            of[i][j] = Activator(doPerceptron(&of[i-1][0], &net->layer[i][j]), net);

    // binary classifier output layer
    return tanhf(doPerceptron(&of[net->num_layers-2][0], &net->layer[net->num_layers-1][0]));
}

float trainNetwork(network* net, const float* inputs, const float target)
{
    // validate [it's ok, the output should be sigmoid 0-1 otherwise]
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(net->layer == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    
/**************************************
    Forward Prop
**************************************/

    // outputs per layer / unit
    float of[net->num_layers-1][net->num_layerunits];
    float output = 0;

    // because now we only have the derivative with respect to x and not f(x).
    float df[net->num_layers-1][net->num_layerunits];

    // input layer
    for(int i = 0; i < net->num_layerunits; i++)
    {
        of[0][i] = doPerceptron(inputs, &net->layer[0][i]);
        df[0][i] = Activator(of[0][i], net);
    }

    // hidden layers
    for(int i = 1; i < net->num_layers-1; i++)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            of[i][j] = doPerceptron(&df[i-1][0], &net->layer[i][j]);
            df[i][j] = Activator(of[i][j], net);
        }
    }

    // binary classifier output layer
    output = tanhf(doPerceptron(&df[net->num_layers-2][0], &net->layer[net->num_layers-1][0]));

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
        ef[net->num_layers-2][i] = net->gain * Derivative(net->output[net->num_layers-2][i], net) * ler;

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
            ef[i][j] = net->gain * Derivative(net->output[i][j], net) * ler;
    }

/**************************************
    Update Weights
**************************************/
    
    // update input layer weights
    for(int j = 0; j < net->num_layerunits; j++)
    {
        uint dt = 0;
        if(net->dropout != 0 && uRandFloat(0.f, 1.f) <= net->dropout)
        {
            if(net->activator != SELU)
                continue;
            dt = 1;
        }
        else if(net->wdropout != 0)
            dt = 2;
        
        for(int k = 0; k < net->layer[0][j].weights; k++)
            net->layer[0][j].data[k] += doDropout(net, Optimiser(net, inputs[k], ef[0][j], &net->layer[0][j].momentum[k]), dt);

        net->layer[0][j].bias += doDropout(net, Optimiser(net, 1, ef[0][j], &net->layer[0][j].bias_momentum), dt);
    }

    // update hidden layer weights
    for(int i = 1; i < net->num_layers-1; i++)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            uint dt = 0;
            if(net->dropout != 0 && uRandFloat(0.f, 1.f) <= net->dropout)
            {
                if(net->activator != SELU)
                    continue;
                dt = 1;
            }
            else if(net->wdropout != 0)
                dt = 2;
            
            for(int k = 0; k < net->layer[i][j].weights; k++)
                net->layer[i][j].data[k] += doDropout(net, Optimiser(net, net->output[i-1][j], ef[i][j], &net->layer[i][j].momentum[k]), dt);

            net->layer[i][j].bias += doDropout(net, Optimiser(net, 1, ef[i][j], &net->layer[i][j].bias_momentum), dt);
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
    net->foutput = 0;
    net->error = 0;
    
    // init weight
    float d = 1.f; //WEIGHT_INIT_RANDOM
    if(net->init == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtps(6.0f/(net->num_inputs+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrtps(3.0f/net->num_inputs);
    else if(net->init == WEIGHT_INIT_UNIFORM_LECUN_POW)
        d = powf(net->num_inputs, 0.5f);
    else if(net->init == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrtps(2.0f/(net->num_inputs+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_NORMAL_LECUN)
        d = sqrtps(1.0f/net->num_inputs);

    // reset first layer perceptrons
    for(int i = 0; i < net->num_layerunits; i++)
        resetPerceptron(&net->layer[0][i], d, net->init);
    
    // weight init
    if(net->init == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtps(6.0f/(net->num_layerunits+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrtps(3.0f/net->num_layerunits);
    else if(net->init == WEIGHT_INIT_UNIFORM_LECUN_POW || net->init == WEIGHT_INIT_NORMAL_LECUN_POW)
        d = powf(net->num_layerunits, 0.5f);
    else if(net->init == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrtps(2.0f/(net->num_layerunits+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_NORMAL_LECUN)
        d = sqrtps(1.0f/net->num_layerunits);

    // reset hidden layers
    for(uint i = 1; i < net->num_layers-1; i++)
        for(int j = 0; j < net->num_layerunits; j++)
            resetPerceptron(&net->layer[i][j], d, net->init);

    // weight init
    if(net->init == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtps(6.0f/(net->num_layerunits+1));
    else if(net->init == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrtps(2.0f/(net->num_layerunits+1));

    // reset output layer
    resetPerceptron(&net->layer[net->num_layers-1][0], d, net->init);
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

int saveNetwork(network* net, const char* file)
{
    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(net->layer == NULL)
        return ERROR_UNINITIALISED_NETWORK;

    FILE* f = fopen(file, "wb");
    if(f != NULL)
    {
        if(flock(fileno(f), LOCK_EX) == -1)
        {
            fclose(f);
            return -1;
        }

        if(fwrite(&net->num_layerunits, 1, sizeof(uint), f) != sizeof(uint))
        {
            fclose(f);
            return -1;
        }
        
        if(fwrite(&net->num_inputs, 1, sizeof(uint), f) != sizeof(uint))
        {
            fclose(f);
            return -1;
        }
        
        if(fwrite(&net->num_layers, 1, sizeof(uint), f) != sizeof(uint))
        {
            fclose(f);
            return -1;
        }
        
        if(fwrite(&net->init, 1, sizeof(uint), f) != sizeof(uint))
        {
            fclose(f);
            return -1;
        }

        if(fwrite(&net->activator, 1, sizeof(uint), f) != sizeof(uint))
        {
            fclose(f);
            return -1;
        }

        if(fwrite(&net->optimiser, 1, sizeof(uint), f) != sizeof(uint))
        {
            fclose(f);
            return -1;
        }

        if(fwrite(&net->batches, 1, sizeof(uint), f) != sizeof(uint))
        {
            fclose(f);
            return -1;
        }

        ///

        if(fwrite(&net->rate, 1, sizeof(float), f) != sizeof(float))
        {
            fclose(f);
            return -1;
        }

        if(fwrite(&net->gain, 1, sizeof(float), f) != sizeof(float))
        {
            fclose(f);
            return -1;
        }

        if(fwrite(&net->dropout, 1, sizeof(float), f) != sizeof(float))
        {
            fclose(f);
            return -1;
        }

        if(fwrite(&net->wdropout, 1, sizeof(float), f) != sizeof(float))
        {
            fclose(f);
            return -1;
        }

        if(fwrite(&net->momentum, 1, sizeof(float), f) != sizeof(float))
        {
            fclose(f);
            return -1;
        }

        if(fwrite(&net->rmsalpha, 1, sizeof(float), f) != sizeof(float))
        {
            fclose(f);
            return -1;
        }
        
        if(fwrite(&net->epsilon, 1, sizeof(float), f) != sizeof(float))
        {
            fclose(f);
            return -1;
        }

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

        if(flock(fileno(f), LOCK_UN) == -1)
        {
            fclose(f);
            return -1;
        }

        fclose(f);
    }

    return 0;
}

int loadNetwork(network* net, const char* file)
{
    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(net->layer == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    
    FILE* f = fopen(file, "rb");
    if(f == NULL)
        return -1;

    ///

    destroyNetwork(net);

    ///

    if(fread(&net->num_layerunits, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -1;
    }

    if(fread(&net->num_inputs, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -1;
    }

    if(fread(&net->num_layers, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -1;
    }

    if(fread(&net->init, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -1;
    }

    if(fread(&net->activator, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -1;
    }

    if(fread(&net->optimiser, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -1;
    }

    if(fread(&net->batches, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -1;
    }

    ///

    if(fread(&net->rate, 1, sizeof(float), f) != sizeof(float))
    {
        fclose(f);
        return -1;
    }

    if(fread(&net->gain, 1, sizeof(float), f) != sizeof(float))
    {
        fclose(f);
        return -1;
    }

    if(fread(&net->dropout, 1, sizeof(float), f) != sizeof(float))
    {
        fclose(f);
        return -1;
    }

    if(fread(&net->wdropout, 1, sizeof(float), f) != sizeof(float))
    {
        fclose(f);
        return -1;
    }

    if(fread(&net->momentum, 1, sizeof(float), f) != sizeof(float))
    {
        fclose(f);
        return -1;
    }

    if(fread(&net->rmsalpha, 1, sizeof(float), f) != sizeof(float))
    {
        fclose(f);
        return -1;
    }
    
    if(fread(&net->epsilon, 1, sizeof(float), f) != sizeof(float))
    {
        fclose(f);
        return -1;
    }

    ///

    createNetwork(net, net->init, net->num_inputs, net->num_layers-2, net->num_layerunits, 0);

    ///

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

    ///

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

    ///

    fclose(f);
    return 0;
}

#endif
