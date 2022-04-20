/*
    James William Fletcher (james@voxdsp.com)
        April 2022

    Info:
    
        PoryDrive, a simple 3D driving game.


    Keyboard:

        ESCAPE = Focus/Unfocus Mouse Look
        F = FPS to console
        P = Player stats to console
        N = New Game
        W = Drive Forward
        A = Turn Left
        S = Drive Backward
        D = Turn Right
        Space = Breaks


    Mouse:

        Scroll = Zoom in/out

    
    Notes:

        Although the 3D model is sourced ready made from a third-party I cannot
        stress enough how much work I had to put in to vertex colouring the segments,
        deleting non-visible faces and generally just cleaning up parts of the mesh
        before and after triangulating it. It was a few hours of work, more finicky
        than anything.

        It made me think that it would be nice to have a program that automatically
        deleted non-visible faces for me, for example, so a render pass from many
        angles across a sphere like peeling an orange top to bottom with a fixed
        increment step. Even still, you get a similar effect in Blender by selecting
        visible faces from multiple angles until you feel like you have all the
        important ones, then you can hide them and delete what was hiding behind them
        or just invert the selection and delete. But the mesh ideally still needs
        some minor cleaning after the fact.

        The windows are still double layered, it's probably not necesary but I left
        it that way.

        This game really is about that E34 car model from you know who automobile
        manafacturer, eveything else is well, garnish.

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <sys/time.h>

#define uint GLushort
#define sint GLshort
#define f32 GLfloat

#include "gl.h"
#define GLFW_INCLUDE_NONE
#include "glfw3.h"

#ifndef __x86_64__ 
    #define NOSSE
#endif

// uncommenting this define will enable the MMX random when using fRandFloat (it's a marginally slower)
#define SEIR_RAND

#include "esAux2.h"

#include "res.h"
#include "assets/bluecube.h"
#include "assets/purplecube.h"
#include "assets/porygon.h"
#include "assets/dna.h"
#include "assets/body.h"
#include "assets/windows.h"
#include "assets/wheel.h"

//*************************************
// globals
//*************************************
GLFWwindow* window;
uint winw = 1024;
uint winh = 768;
double t = 0;   // time
double dt = 0;  // delta time
double fc = 0;  // frame count
double lfct = 0;// last frame count time
f32 aspect;
double x,y,lx,ly;
double rww, ww, rwh, wh, ww2, wh2;
double uw, uh, uw2, uh2; // normalised pixel dpi

// render state id's
GLint projection_id;
GLint modelview_id;
GLint position_id;
GLint lightpos_id;
GLint solidcolor_id;
GLint color_id;
GLint opacity_id;
GLint normal_id;

// render state matrices
mat projection;
mat view;
mat model;
mat modelview;
mat viewrot;

// render state inputs
vec lightpos = {0.f, 0.f, 0.f};

// models
sint bindstate = -1;
uint keystate[6] = {0};
ESModel mdlBlueCube;
ESModel mdlPurpleCube;
ESModel mdlPorygon;
ESModel mdlDNA;
ESModel mdlBody;
ESModel mdlWindows;
ESModel mdlWheel;

// game vars
#define FAR_DISTANCE 1000.f
#define NEWGAME_SEED 1337

// camera vars
uint focus_cursor = 1;
double sens = 0.001f;
f32 xrot = 2.6f;
f32 yrot = 1.7f;
f32 zoom = -0.3f;

// player vars
uint pg;// gear
f32 pr; // rotation
f32 sr; // steering rotation
vec pp; // position
vec pv; // velocity
vec pd; // direction
f32 sp; // speed
uint cp;// collected porygon count
double st=0; // start time
char tts[32];// time taken string

// porygon vars
vec zp; // position
vec zd; // direction
f32 zr; // rotation
f32 zs; // speed
double za;// alive state
f32 zt; // twitch radius

// configurable vars
f32 maxspeed = 0.0095f;
f32 acceleration = 0.0025f;
f32 inertia = 0.00015f;
f32 drag = 0.00038f;
f32 steeringspeed = 1.2f;
f32 steerinertia = 233.f;
f32 minsteer = 0.32f;
f32 maxsteer = 0.55f;
f32 steeringtransfer = 0.023f;
f32 steeringtransferinertia = 280.f;

//*************************************
// utility functions
//*************************************
void loadConfig()
{
    FILE* f = fopen("config.txt", "r");
    if(f)
    {
        printf("\nDetected config.txt loading settings...\n");
        char line[256];
        while(fgets(line, 256, f) != NULL)
        {
            char set[64];
            memset(set, 0, 64);
            float val;
            
            if(sscanf(line, "%63s %f", set, &val) == 2)
            {
                printf("Setting Loaded: %s %g\n", set, val);

                if(strcmp(set, "maxspeed") == 0){maxspeed = val;}
                if(strcmp(set, "acceleration") == 0){acceleration = val;}
                if(strcmp(set, "inertia") == 0){inertia = val;}
                if(strcmp(set, "drag") == 0){drag = val;}
                if(strcmp(set, "steeringspeed") == 0){steeringspeed = val;}
                if(strcmp(set, "steerinertia") == 0){steerinertia = val;}
                if(strcmp(set, "minsteer") == 0){minsteer = val;}
                if(strcmp(set, "maxsteer") == 0){maxsteer = val;}
                if(strcmp(set, "steeringtransfer") == 0){steeringtransfer = val;}
                if(strcmp(set, "steeringtransferinertia") == 0){steeringtransferinertia = val;}
            }
        }
        fclose(f);
    }
}

void timestamp(char* ts)
{
    const time_t tt = time(0);
    strftime(ts, 16, "%H:%M:%S", localtime(&tt));
}

static inline f32 fRandFloat(const float min, const float max)
{
    return min + randf() * (max-min); 
}

void timeTaken(uint ss)
{
    if(ss == 1)
    {
        const double tt = t-st;
        if(tt < 60.0)
            sprintf(tts, "%.2f Sec", tt);
        else if(tt < 3600.0)
            sprintf(tts, "%.2f Min", tt * 0.016666667);
        else if(tt < 216000.0)
            sprintf(tts, "%.2f Hr", tt * 0.000277778);
        else if(tt < 12960000.0)
            sprintf(tts, "%.2f Days", tt * 0.00000463);
    }
    else
    {
        const double tt = t-st;
        if(tt < 60.0)
            sprintf(tts, "%.2f Seconds", tt);
        else if(tt < 3600.0)
            sprintf(tts, "%.2f Minutes", tt * 0.016666667);
        else if(tt < 216000.0)
            sprintf(tts, "%.2f Hours", tt * 0.000277778);
        else if(tt < 12960000.0)
            sprintf(tts, "%.2f Days", tt * 0.00000463);
    }
}

//*************************************
// render functions
//*************************************

void iterDNA()
{
    // so that you don't get confused I am exploiting the
    // fact that when a colour channel goes into minus
    // figures very strange things begin to happen to the
    // gradient between triangle verticies.
    static const uint mi = dna_numvert*3;
    static f32 cd = 1.f;
    for(uint i = 0; i < mi; i++) // lavalamp it
    {
        dna_colors[i] += fRandFloat(0.1f, 0.6f) * cd;

        // and this piece of code prevents it looking like a random mess,
        // gives some structure. This is the lava lamper.
        if(dna_colors[i] >= 1.f)
            cd = -1.f;
        else if(dna_colors[i] <= 0.f)
            cd = 1.f;
    }
    esRebind(GL_ARRAY_BUFFER, &mdlDNA.cid, dna_colors, sizeof(dna_colors), GL_STATIC_DRAW);
}

void rCube(f32 x, f32 y)
{
    mIdent(&model);
    mTranslate(&model, x, y, 0.f);
    mMul(&modelview, &model, &view);

    glUniform1f(opacity_id, 1.0f);

    glUniformMatrix4fv(projection_id, 1, GL_FALSE, (f32*) &projection.m[0][0]);
    glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
    
    if(bindstate != 1)
    {
        glBindBuffer(GL_ARRAY_BUFFER, mdlPurpleCube.cid);
        glVertexAttribPointer(color_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(color_id);

        glBindBuffer(GL_ARRAY_BUFFER, mdlPurpleCube.vid);
        glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(position_id);

        glBindBuffer(GL_ARRAY_BUFFER, mdlPurpleCube.nid);
        glVertexAttribPointer(normal_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(normal_id);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdlPurpleCube.iid);
        
        bindstate = 1;
    }

    const f32 dlap = vDistLa(zp, (vec){x, y, 0.f}); // porygon
    if(dlap < 0.15f)
    {
        vec nf;
        vSub(&nf, zp, (vec){x, y, 0.f});
        vNorm(&nf);
        vMulS(&nf, nf, 0.15f-dlap);
        vAdd(&zp, zp, nf);
    }
    const f32 dla = vDist(pp, (vec){x, y, 0.f}); // car
    if(dla < 0.15f)
    {
        vec nf;
        vSub(&nf, pp, (vec){x, y, 0.f});
        vNorm(&nf);
        vMulS(&nf, nf, fabsf(sp)); //dla*sp*10.f
        vAdd(&pv, pv, nf);
    }
    if((dla < 0.16f || dlap < 0.16f) && bindstate != 2)
    {
        glBindBuffer(GL_ARRAY_BUFFER, mdlBlueCube.cid);
        glVertexAttribPointer(color_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(color_id);

        glBindBuffer(GL_ARRAY_BUFFER, mdlBlueCube.vid);
        glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(position_id);

        glBindBuffer(GL_ARRAY_BUFFER, mdlBlueCube.nid);
        glVertexAttribPointer(normal_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(normal_id);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdlBlueCube.iid);

        bindstate = 2;
    }

    glDrawElements(GL_TRIANGLES, purplecube_numind, GL_UNSIGNED_SHORT, 0);
}

void rPorygon(f32 x, f32 y, f32 r)
{
    bindstate = -1;

    mIdent(&model);
    mTranslate(&model, x, y, 0.f);
    mRotZ(&model, r);

    if(za != 0.0)
        mScale(&model, 1.f, 1.f, 0.1f);

    mMul(&modelview, &model, &view);

    // returns direction
    mGetDirY(&zd, model);
    vInv(&zd);

    if(za != 0.0)
        glUniform1f(opacity_id, (za-t)/6.0);
    else
        glUniform1f(opacity_id, 1.0f);

    glUniformMatrix4fv(projection_id, 1, GL_FALSE, (f32*) &projection.m[0][0]);
    glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
    
    glBindBuffer(GL_ARRAY_BUFFER, mdlPorygon.cid);
    glVertexAttribPointer(color_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(color_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdlPorygon.vid);
    glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(position_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdlPorygon.nid);
    glVertexAttribPointer(normal_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(normal_id);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdlPorygon.iid);

    if(za != 0.0)
        glEnable(GL_BLEND);
    glDrawElements(GL_TRIANGLES, porygon_numind, GL_UNSIGNED_SHORT, 0);
    if(za != 0.0)
        glDisable(GL_BLEND);
}

void rDNA(f32 x, f32 y, f32 z)
{
    bindstate = -1;

    static f32 dr = 0.f;
    dr += 1.f * dt;

    mIdent(&model);
    mTranslate(&model, x, y, z);
    mRotZ(&model, dr);
    mMul(&modelview, &model, &view);

    glUniform1f(opacity_id, 1.0f);

    glUniformMatrix4fv(projection_id, 1, GL_FALSE, (f32*) &projection.m[0][0]);
    glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
    
    glBindBuffer(GL_ARRAY_BUFFER, mdlDNA.cid);
    glVertexAttribPointer(color_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(color_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdlDNA.vid);
    glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(position_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdlDNA.nid);
    glVertexAttribPointer(normal_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(normal_id);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdlDNA.iid);

    glDrawElements(GL_TRIANGLES, dna_numind, GL_UNSIGNED_SHORT, 0);
}

void rCar(f32 x, f32 y, f32 z, f32 rx)
{
    bindstate = -1;

    // opaque
    glUniform1f(opacity_id, 1.0f);

    // wheel spin speed
    static f32 wr = 0.f;
    const f32 speed = sp * 33.f;
    if(pg == 0)
        wr += speed;
    else
        wr -= speed;

    // wheel; front left
    mIdent(&model);
    mTranslate(&model, x, y, z);
    mRotZ(&model, -rx);
    mTranslate(&model, 0.026343f, -0.054417f, 0.012185f);
    mRotZ(&model, sr);

    // returns direction
    mGetDirY(&pd, model);
    vInv(&pd);

    mRotY(&model, -wr);
    mMul(&modelview, &model, &view);

    glUniformMatrix4fv(projection_id, 1, GL_FALSE, (f32*) &projection.m[0][0]);
    glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
    
    glBindBuffer(GL_ARRAY_BUFFER, mdlWheel.cid);
    glVertexAttribPointer(color_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(color_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdlWheel.vid);
    glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(position_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdlWheel.nid);
    glVertexAttribPointer(normal_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(normal_id);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdlWheel.iid);

    glDrawElements(GL_TRIANGLES, wheel_numind, GL_UNSIGNED_SHORT, 0);

    // wheel; back left

    mIdent(&model);
    mTranslate(&model, x, y, z);
    mRotZ(&model, -rx);
    mTranslate(&model, 0.026343f, 0.045294f, 0.012185f);
    mRotY(&model, -wr);
    mMul(&modelview, &model, &view);

    glUniformMatrix4fv(projection_id, 1, GL_FALSE, (f32*) &projection.m[0][0]);
    glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
    
    glBindBuffer(GL_ARRAY_BUFFER, mdlWheel.cid);
    glVertexAttribPointer(color_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(color_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdlWheel.vid);
    glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(position_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdlWheel.nid);
    glVertexAttribPointer(normal_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(normal_id);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdlWheel.iid);

    glDrawElements(GL_TRIANGLES, wheel_numind, GL_UNSIGNED_SHORT, 0);

    // wheel; front right

    mIdent(&model);
    mRotZ(&model, PI);
    mTranslate(&model, -x, -y, -z);
    mRotZ(&model, -rx);
    mTranslate(&model, 0.026343f, 0.054417f, 0.012185f);
    mRotZ(&model, sr);
    mRotY(&model, wr);
    mMul(&modelview, &model, &view);

    glUniformMatrix4fv(projection_id, 1, GL_FALSE, (f32*) &projection.m[0][0]);
    glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
    
    glBindBuffer(GL_ARRAY_BUFFER, mdlWheel.cid);
    glVertexAttribPointer(color_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(color_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdlWheel.vid);
    glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(position_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdlWheel.nid);
    glVertexAttribPointer(normal_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(normal_id);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdlWheel.iid);

    glDrawElements(GL_TRIANGLES, wheel_numind, GL_UNSIGNED_SHORT, 0);

    // wheel; back right

    mIdent(&model);
    mRotZ(&model, PI);
    mTranslate(&model, -x, -y, -z);
    mRotZ(&model, -rx);
    mTranslate(&model, 0.026343f, -0.045294f, 0.012185f);
    mRotY(&model, wr);
    mMul(&modelview, &model, &view);

    glUniformMatrix4fv(projection_id, 1, GL_FALSE, (f32*) &projection.m[0][0]);
    glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
    
    glBindBuffer(GL_ARRAY_BUFFER, mdlWheel.cid);
    glVertexAttribPointer(color_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(color_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdlWheel.vid);
    glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(position_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdlWheel.nid);
    glVertexAttribPointer(normal_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(normal_id);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdlWheel.iid);

    glDrawElements(GL_TRIANGLES, wheel_numind, GL_UNSIGNED_SHORT, 0);

    // body & window matrix

    mIdent(&model);
    mTranslate(&model, x, y, z);
    mRotZ(&model, -rx);
    mMul(&modelview, &model, &view);

    glUniformMatrix4fv(projection_id, 1, GL_FALSE, (f32*) &projection.m[0][0]);
    glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
    
    // body

    glBindBuffer(GL_ARRAY_BUFFER, mdlBody.cid);
    glVertexAttribPointer(color_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(color_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdlBody.vid);
    glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(position_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdlBody.nid);
    glVertexAttribPointer(normal_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(normal_id);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdlBody.iid);

    glDisable(GL_CULL_FACE);
    glDrawElements(GL_TRIANGLES, body_numind, GL_UNSIGNED_SHORT, 0);
    glEnable(GL_CULL_FACE);

    // transparent
    glUniform1f(opacity_id, 0.3f);

    // windows

    glBindBuffer(GL_ARRAY_BUFFER, mdlWindows.cid);
    glVertexAttribPointer(color_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(color_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdlWindows.vid);
    glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(position_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdlWindows.nid);
    glVertexAttribPointer(normal_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(normal_id);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdlWindows.iid);

    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);
    glDrawElements(GL_TRIANGLES, windows_numind, GL_UNSIGNED_SHORT, 0);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
}

//*************************************
// game functions
//*************************************
void newGame(unsigned int seed)
{
    srand(seed);
    srandf(seed);

    loadConfig();

    char strts[16];
    timestamp(&strts[0]);
    printf("\n[%s] Game Start [%u].\n", strts, seed);
    
    glfwSetWindowTitle(window, "PoryDrive");
    
    pp = (vec){0.f, 0.f, 0.f};
    pv = (vec){0.f, 0.f, 0.f};
    pd = (vec){0.f, 0.f, 0.f};

    st = 0;

    cp = 0;
    pg = 0;
    pr = 0.f;
    sr = 0.f;
    sp = 0.f;

    zp = (vec){fRandFloat(-18.f, 18.f), fRandFloat(-18.f, 18.f), 0.f};
    //zp = (vec){0.f, 0.3f, 0.f};
    zs = 0.3f;
    za = 0.0;
    zt = 8.f;
}

//*************************************
// update & render
//*************************************
void main_loop()
{
//*************************************
// time delta for interpolation
//*************************************
    static double lt = 0;
    dt = t-lt;
    lt = t;

//*************************************
// keystates
//*************************************
    f32 tr = maxsteer * ((maxspeed-sp) * steerinertia);
    if(tr < minsteer){tr = minsteer;}

    if(keystate[0] == 1)
    {
        sr -= steeringspeed * dt;
        if(sr < -tr){sr = -tr;}
    }

    if(keystate[1] == 1)
    {
        sr += steeringspeed * dt;
        if(sr > tr){sr = tr;}
    }

    if(keystate[0] == 0 && keystate[1] == 0)
    {
        if(sr > 0.001f)
            sr -= steeringspeed * dt;
        else if(sr < -0.001f)
            sr += steeringspeed * dt;
    }
    
    if(keystate[2] == 1)
    {
        vec inc;
        sp += acceleration * dt;
        vMulS(&inc, pd, acceleration * dt);
        vAdd(&pv, pv, inc);
        pg = 0;
    }

    if(keystate[3] == 1)
    {    
        vec inc;
        sp -= acceleration * dt;
        vMulS(&inc, pd, -acceleration * dt);
        vAdd(&pv, pv, inc);
        pg = 1;
    }

    if(keystate[4] == 1)
    {
        sp *= 0.99f * (1.f-dt);
    }

//*************************************
// update title bar stats
//*************************************
    static double ltut = 3.0;
    if(t > ltut)
    {
        timeTaken(1);
        char title[256];
        const f32 dsp = fabsf(sp*(1.f/maxspeed)*130.f);
        sprintf(title, "| %s | Speed %.f MPH | Porygon %u |", tts, dsp, cp);
        sprintf(title, "| %s | Speed %.f MPH | Porygon %u |", tts, dsp, cp);
        glfwSetWindowTitle(window, title);
        ltut = t + 1.0;
    }

//*************************************
// simulate car
//*************************************

    if(sp > 0.f)
        sp -= drag * dt;
    else
        sp += drag * dt;

    if(fabsf(sp) > maxspeed)
    {
        if(pg == 0)
            sp = maxspeed;
        else
            sp = -maxspeed;
    }

    if(sp > inertia || sp < -inertia)
    {
        vAdd(&pp, pp, pv);
        vMulS(&pv, pd, sp);
        pr -= sr * steeringtransfer * (sp*steeringtransferinertia);
    }

    if(pp.x > 17.5f){pp.x = 17.5f;}
    else if(pp.x < -17.5f){pp.x = -17.5f;}
    if(pp.y > 17.5f){pp.y = 17.5f;}
    else if(pp.y < -17.5f){pp.y = -17.5f;}

//*************************************
// simulate porygon
//*************************************

    if(za == 0.0)
    {
        vec inc;
        vMulS(&inc, zd, zs * dt);
        vAdd(&zp, zp, inc);
        zr += fRandFloat(-zt, zt) * dt;

        if(zp.x > 17.5f){zp.x = 17.5f; zr = fRandFloat(-PI, PI);}
        else if(zp.x < -17.5f){zp.x = -17.5f; zr = fRandFloat(-PI, PI);}
        if(zp.y > 17.5f){zp.y = 17.5f; zr = fRandFloat(-PI, PI);}
        else if(zp.y < -17.5f){zp.y = -17.5f; zr = fRandFloat(-PI, PI);}

        if(vDist(pp, zp) < 0.1f)
        {
            cp++;
            za = t+6.0;
            iterDNA();
        }
    }
    else if(t > za)
    {
        zp = (vec){fRandFloat(-18.f, 18.f), fRandFloat(-18.f, 18.f), 0.f};
        zs = fRandFloat(0.3f, 1.f);
        zt = fRandFloat(8.f, 16.f);
        za = 0.0;
    }

//*************************************
// camera
//*************************************

    if(focus_cursor == 1)
    {
        glfwGetCursorPos(window, &x, &y);

        xrot += (ww2-x)*sens;
        yrot += (wh2-y)*sens;

        if(yrot > 1.5f)
            yrot = 1.5f;
        if(yrot < 0.5f)
            yrot = 0.5f;

        glfwSetCursorPos(window, ww2, wh2);
    }

    mIdent(&view);
    mTranslate(&view, 0.f, -0.033f, zoom);
    mRotate(&view, yrot, 1.f, 0.f, 0.f);
    mRotate(&view, xrot, 0.f, 0.f, 1.f);
    mTranslate(&view, -pp.x, -pp.y, -pp.z);

//*************************************
// begin render
//*************************************
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//*************************************
// main render
//*************************************

    // render scene
    for(f32 i = -17.5f; i <= 18.f; i += 0.53f)
        for(f32 j = -17.5f; j <= 18.f; j += 0.53f)
            if((i < -0.1f || i > 0.1f) || (j < -0.1f || j > 0.1f)) // lol a branch for so little tut tut
                rCube(i, j);

    // render porygon
    rPorygon(zp.x, zp.y, zr);

    // render dna
    rDNA(0.f, 0.f, 0.1f);

    // render player
    shadeLambert3(&position_id, &projection_id, &modelview_id, &lightpos_id, &normal_id, &color_id, &opacity_id);
    glUniform3f(lightpos_id, lightpos.x, lightpos.y, lightpos.z);
    rCar(pp.x, pp.y, pp.z, pr);


//*************************************
// swap buffers / display render
//*************************************
    glfwSwapBuffers(window);
}

//*************************************
// Input Handelling
//*************************************
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // control
    if(action == GLFW_PRESS)
    {
        if(key == GLFW_KEY_A){ keystate[0] = 1; }
        else if(key == GLFW_KEY_D){ keystate[1] = 1; }
        else if(key == GLFW_KEY_W){ keystate[2] = 1; }
        else if(key == GLFW_KEY_S){ keystate[3] = 1; }
        else if(key == GLFW_KEY_SPACE){ keystate[4] = 1; }

        // new game
        else if(key == GLFW_KEY_N)
        {
            // end
            timeTaken(0);
            char strts[16];
            timestamp(&strts[0]);
            printf("[%s] Porygon Collected: %u\n", strts, cp);
            printf("[%s] Time-Taken: %s or %g Seconds\n", strts, tts, t-st);
            printf("[%s] Game End.\n", strts);
            
            // new
            newGame(time(0));
        }

        // else if(key == GLFW_KEY_R)
        // {
        //     printf("%g %g %g\n", dna_colors[0], dna_colors[1], dna_colors[2]);
        //     iterDNA();
        //     cp++;
        // }

        // stats
        else if(key == GLFW_KEY_P)
        {
            char strts[16];
            timestamp(&strts[0]);
            printf("[%s] Porygon Collected: %u\n", strts, cp);
        }

        // toggle mouse focus
        if(key == GLFW_KEY_ESCAPE)
        {
            focus_cursor = 1 - focus_cursor;
            if(focus_cursor == 0)
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            else
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
            glfwSetCursorPos(window, ww2, wh2);
            glfwGetCursorPos(window, &ww2, &wh2);
        }
    }
    else if(action == GLFW_RELEASE)
    {
        if(key == GLFW_KEY_A){ keystate[0] = 0; }
        else if(key == GLFW_KEY_D){ keystate[1] = 0; }
        else if(key == GLFW_KEY_W){ keystate[2] = 0; }
        else if(key == GLFW_KEY_S){ keystate[3] = 0; }
        else if(key == GLFW_KEY_SPACE){ keystate[4] = 0; }
    }

    // show average fps
    if(key == GLFW_KEY_F)
    {
        if(t-lfct > 2.0)
        {
            char strts[16];
            timestamp(&strts[0]);
            printf("[%s] FPS: %g\n", strts, fc/(t-lfct));
            lfct = t;
            fc = 0;
        }
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if(yoffset == -1)
        zoom += 0.06f * zoom;
    else if(yoffset == 1)
        zoom -= 0.06f * zoom;
    
    if(zoom > -0.11f){zoom = -0.11f;}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if(action == GLFW_PRESS)
    {
        if(button == GLFW_MOUSE_BUTTON_4 || button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            if(zoom != -0.3f)
                zoom = -0.3f;
            else
                zoom = -3.3f;
        }
    }
}

void window_size_callback(GLFWwindow* window, int width, int height)
{
    winw = width;
    winh = height;

    glViewport(0, 0, winw, winh);
    aspect = (f32)winw / (f32)winh;
    ww = winw;
    wh = winh;
    rww = 1/ww;
    rwh = 1/wh;
    ww2 = ww/2;
    wh2 = wh/2;
    uw = (double)aspect / ww;
    uh = 1 / wh;
    uw2 = (double)aspect / ww2;
    uh2 = 1 / wh2;

    mIdent(&projection);
    mPerspective(&projection, 60.0f, aspect, 0.01f, FAR_DISTANCE*2.f); 
}

//*************************************
// Process Entry Point
//*************************************
int main(int argc, char** argv)
{
    // allow custom msaa level
    int msaa = 16;
    if(argc >= 2){msaa = atoi(argv[1]);}

    // help
    printf("----\n");
    printf("PoryDrive\n");
    printf("----\n");
    printf("James William Fletcher (james@voxdsp.com)\n");
    printf("----\n");
    printf("There is only one command line argument, and that is the MSAA level 0-16.\n");
    printf("----\n");
    printf("~ Keyboard Input:\n");
    printf("ESCAPE = Focus/Unfocus Mouse Look\n");
    printf("F = FPS to console\n");
    printf("P = Player stats to console\n");
    printf("N = New Game\n");
    printf("W = Drive Forward\n");
    printf("A = Turn Left\n");
    printf("S = Drive Backward\n");
    printf("D = Turn Right\n");
    printf("Space = Break\n");
    printf("----\n");
    printf("~ Mouse Input:\n");
    printf("RIGHT/MOUSE4 = Zoom Snap Close/Ariel\n");
    printf("Scroll = Zoom in/out\n");
    printf("----\n");
    printf("~ How to play:\n");
    printf("Drive around and \"collect\" Porygon, each time you collect a Porygon a new one will randomly spawn somewhere on the map. A Porygon colliding with a purple cube will cause it to light up blue, this can help you find them. Upon right clicking the mouse you will switch between Ariel and Close views, in the Ariel view it is easier to see which of the purple cubes that the Porygon is colliding with.\n");
    printf("----\n");

    // init glfw
    if(!glfwInit()){exit(EXIT_FAILURE);}
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_SAMPLES, msaa);
    window = glfwCreateWindow(winw, winh, "PoryDrive", NULL, NULL);
    if(!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    const GLFWvidmode* desktop = glfwGetVideoMode(glfwGetPrimaryMonitor());
    glfwSetWindowPos(window, (desktop->width/2)-(winw/2), (desktop->height/2)-(winh/2)); // center window on desktop
    glfwSetWindowSizeCallback(window, window_size_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwMakeContextCurrent(window);
    gladLoadGL(glfwGetProcAddress);
    glfwSwapInterval(1); // 0 for immediate updates, 1 for updates synchronized with the vertical retrace, -1 for adaptive vsync

    // set icon
    glfwSetWindowIcon(window, 1, &(GLFWimage){16, 16, (unsigned char*)&icon_image.pixel_data});

    // hide cursor
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

    // reset mouse for camera rotation
    glfwSetCursorPos(window, ww2, wh2);
    glfwGetCursorPos(window, &ww2, &wh2);

//*************************************
// projection
//*************************************

    window_size_callback(window, winw, winh);

//*************************************
// bind vertex and index buffers
//*************************************

    // ***** BIND BODY *****
    esBind(GL_ARRAY_BUFFER, &mdlBody.vid, body_vertices, sizeof(body_vertices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlBody.nid, body_normals, sizeof(body_normals), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlBody.iid, body_indices, sizeof(body_indices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlBody.cid, body_colors, sizeof(body_colors), GL_STATIC_DRAW);

    // ***** BIND WINDOWS *****
    esBind(GL_ARRAY_BUFFER, &mdlWindows.vid, windows_vertices, sizeof(windows_vertices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlWindows.nid, windows_normals, sizeof(windows_normals), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlWindows.iid, windows_indices, sizeof(windows_indices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlWindows.cid, windows_colors, sizeof(windows_colors), GL_STATIC_DRAW);

    // ***** BIND WHEEL *****
    esBind(GL_ARRAY_BUFFER, &mdlWheel.vid, wheel_vertices, sizeof(wheel_vertices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlWheel.nid, wheel_normals, sizeof(wheel_normals), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlWheel.iid, wheel_indices, sizeof(wheel_indices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlWheel.cid, wheel_colors, sizeof(wheel_colors), GL_STATIC_DRAW);

    // ***** BIND PURPLE CUBE *****
    esBind(GL_ARRAY_BUFFER, &mdlPurpleCube.vid, purplecube_vertices, sizeof(purplecube_vertices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlPurpleCube.nid, purplecube_normals, sizeof(purplecube_normals), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlPurpleCube.iid, purplecube_indices, sizeof(purplecube_indices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlPurpleCube.cid, purplecube_colors, sizeof(purplecube_colors), GL_STATIC_DRAW);

    // ***** BIND BLUE CUBE *****
    esBind(GL_ARRAY_BUFFER, &mdlBlueCube.vid, bluecube_vertices, sizeof(bluecube_vertices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlBlueCube.nid, bluecube_normals, sizeof(bluecube_normals), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlBlueCube.iid, bluecube_indices, sizeof(bluecube_indices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlBlueCube.cid, bluecube_colors, sizeof(bluecube_colors), GL_STATIC_DRAW);

    // ***** BIND PORYGON *****
    esBind(GL_ARRAY_BUFFER, &mdlPorygon.vid, porygon_vertices, sizeof(porygon_vertices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlPorygon.nid, porygon_normals, sizeof(porygon_normals), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlPorygon.iid, porygon_indices, sizeof(porygon_indices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlPorygon.cid, porygon_colors, sizeof(porygon_colors), GL_STATIC_DRAW);

    // ***** BIND DNA *****
    esBind(GL_ARRAY_BUFFER, &mdlDNA.vid, dna_vertices, sizeof(dna_vertices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlDNA.nid, dna_normals, sizeof(dna_normals), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlDNA.iid, dna_indices, sizeof(dna_indices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlDNA.cid, dna_colors, sizeof(dna_colors), GL_STATIC_DRAW);

//*************************************
// compile & link shader programs
//*************************************

    //makeAllShaders();
    makeLambert3();

//*************************************
// configure render options
//*************************************

    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);
    
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.13, 0.13, 0.13, 0.0);

//*************************************
// execute update / render loop
//*************************************

    // init
    newGame(NEWGAME_SEED);

    // reset
    t = glfwGetTime();
    lfct = t;
    
    // event loop
    while(!glfwWindowShouldClose(window))
    {
        t = glfwGetTime();
        glfwPollEvents();
        main_loop();
        fc++;
    }

    // end
    timeTaken(0);
    char strts[16];
    timestamp(&strts[0]);
    printf("[%s] Porygon Collected: %u\n", strts, cp);
    printf("[%s] Time-Taken: %s or %g Seconds\n", strts, tts, t-st);
    printf("[%s] Game End.\n\n", strts);

    // done
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
    return 0;
}
