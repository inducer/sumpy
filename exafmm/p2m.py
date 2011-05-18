from __future__ import division

import numpy as np
from mako.template import Template
from pytools import memoize_method

import pyopencl as cl
import pyopencl.array as cl_array

from exafmm.kernel_common import (
        COMMON_PREAMBLE, FMMParameters)




P2M_KERNEL = Template(
COMMON_PREAMBLE +
    """//CL//
    typedef ${offset_type} offset_t;
    typedef ${mpole_offset_type} mpole_offset_t;
    typedef ${coefficient_type} coeff_t;
    typedef ${geometry_type} geometry_t;
    typedef ${geometry_type}${dimensions} geometry_vec_t;
    typedef ${output_type} output_t;

    __kernel
    __attribute__((reqd_work_group_size()))
    void p2m(
        int *keysGlob, int *rangeGlob, float *targetGlob, float *sourceGlob) 
    {
        int keys = keysGlob[blockIdx.x];
        int numList = rangeGlob[keys];
        float target[2] = {0, 0};
        __shared__ float targetShrd[3];
        __shared__ float sourceShrd[4*THREADS];
        int itarget = blockIdx.x * THREADS;
        targetShrd[0] = targetGlob[6*itarget+0];
        targetShrd[1] = targetGlob[6*itarget+1];
        targetShrd[2] = targetGlob[6*itarget+2];
        for( int ilist=0; ilist<numList; ++ilist ) 
        {
            int begin = rangeGlob[keys+3*ilist+1];
            int size  = rangeGlob[keys+3*ilist+2];
            for( int iblok=0; iblok<(size-1)/THREADS; ++iblok ) 
            {
                int isource = begin + iblok * THREADS + threadIdx.x;
                __syncthreads();
                sourceShrd[4*threadIdx.x+0] = sourceGlob[7*isource+0];
                sourceShrd[4*threadIdx.x+1] = sourceGlob[7*isource+1];
                sourceShrd[4*threadIdx.x+2] = sourceGlob[7*isource+2];
                sourceShrd[4*threadIdx.x+3] = sourceGlob[7*isource+3];
                __syncthreads();
                for( int i=0; i<THREADS; ++i ) 
                {
                    float3 d;
                    d.x = sourceShrd[4*i+0] - targetShrd[0];
                    d.y = sourceShrd[4*i+1] - targetShrd[1];
                    d.z = sourceShrd[4*i+2] - targetShrd[2];
                    float rho,alpha,beta;
                    cart2sph(rho,alpha,beta,d.x,d.y,d.z);
                    LaplaceP2M_core(target,rho,alpha,beta,sourceShrd[4*i+3]);
                }
            }

            int iblok = (size-1)/THREADS;
            int isource = begin + iblok * THREADS + threadIdx.x;
            __syncthreads();
            if( threadIdx.x < size - iblok * THREADS ) 
            {
                sourceShrd[4*threadIdx.x+0] = sourceGlob[7*isource+0];
                sourceShrd[4*threadIdx.x+1] = sourceGlob[7*isource+1];
                sourceShrd[4*threadIdx.x+2] = sourceGlob[7*isource+2];
                sourceShrd[4*threadIdx.x+3] = sourceGlob[7*isource+3];
            }
            __syncthreads();
            for( int i=0; i<size-iblok*THREADS; ++i ) 
            {
                float3 d;
                d.x = sourceShrd[4*i+0] - targetShrd[0];
                d.y = sourceShrd[4*i+1] - targetShrd[1];
                d.z = sourceShrd[4*i+2] - targetShrd[2];
                float rho,alpha,beta;
                cart2sph(rho,alpha,beta,d.x,d.y,d.z);
                LaplaceP2M_core(target,rho,alpha,beta,sourceShrd[4*i+3]);
            }
        }
        itarget = blockIdx.x * THREADS + threadIdx.x;
        targetGlob[6*itarget+0] = target[0];
        targetGlob[6*itarget+1] = target[1];
    }
    """, strict_undefined=True, disable_unicode=True)





# vim: foldmethod=marker filetype=pyopencl.python
