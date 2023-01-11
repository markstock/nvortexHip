/*
 * ngHip05.hip
 *
 * (c)2022 Mark J. Stock <markjstock@gmail.com>
 *
 * v0.5  blocking the cpu calculation for data locality and improved summation accuracy, reordering gpu calls
 */

#include <vector>
#include <random>
#include <chrono>

#include <hip/hip_runtime.h>


// compute using float or double
#define FLOAT float
#define RSQRT rsqrtf

#define CPU_SRC_BLK 256
#define CPU_TRG_BLK 32

// threads per block (hard coded)
#define THREADS_PER_BLOCK 256

// GPU count limit
#define MAX_GPUS 8

// -------------------------
// compute kernel - GPU
__global__ void ngrav_3d_nograds_gpu(
    const int32_t nSrc,
    const FLOAT* const __restrict__ sx,
    const FLOAT* const __restrict__ sy,
    const FLOAT* const __restrict__ sz,
    const FLOAT* const __restrict__ ss,
    const FLOAT* const __restrict__ sr,
    const int32_t tOffset,
    const FLOAT* const __restrict__ tx,
    const FLOAT* const __restrict__ ty,
    const FLOAT* const __restrict__ tz,
    const FLOAT* const __restrict__ tr,
    FLOAT* const __restrict__ tu,
    FLOAT* const __restrict__ tv,
    FLOAT* const __restrict__ tw) {

  // local "thread" id - this is the target particle
  const int32_t i = tOffset + blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;

  // load sources into shared memory (or not)
  __shared__ FLOAT s_sx[THREADS_PER_BLOCK];
  __shared__ FLOAT s_sy[THREADS_PER_BLOCK];
  __shared__ FLOAT s_sz[THREADS_PER_BLOCK];
  __shared__ FLOAT s_ss[THREADS_PER_BLOCK];
  __shared__ FLOAT s_sr[THREADS_PER_BLOCK];

  // velocity accumulators for target point
  FLOAT locu = 0.0f;
  FLOAT locv = 0.0f;
  FLOAT locw = 0.0f;

  FLOAT tr2 = tr[i]*tr[i];

  // which sources do we iterate over?
  const int32_t jcount = nSrc / gridDim.y;
  const int32_t jstart = blockIdx.y * jcount;

  for (int32_t b=0; b<jcount/THREADS_PER_BLOCK; ++b) {
    __syncthreads();

    const int32_t gidx = jstart + b*THREADS_PER_BLOCK + threadIdx.x;
    s_sx[threadIdx.x] = sx[gidx];
    s_sy[threadIdx.x] = sy[gidx];
    s_sz[threadIdx.x] = sz[gidx];
    s_ss[threadIdx.x] = ss[gidx];
    s_sr[threadIdx.x] = sr[gidx];
    __syncthreads();

    // loop over all source points
    // this reduces VGPR use, but does not improve performance
    //#pragma unroll 1
    for (int32_t j=0; j<THREADS_PER_BLOCK; ++j) {
      FLOAT dx = s_sx[j] - tx[i];
      FLOAT dy = s_sy[j] - ty[i];
      FLOAT dz = s_sz[j] - tz[i];
      FLOAT distsq = dx*dx + dy*dy + dz*dz + s_sr[j]*s_sr[j] + tr2;
      FLOAT factor = s_ss[j] * RSQRT(distsq) / distsq;
      locu += dx * factor;
      locv += dy * factor;
      locw += dz * factor;
    }
  }

  // save into device view using atomics
  atomicAdd(&tu[i], locu / (4.0f*3.1415926536f));
  atomicAdd(&tv[i], locv / (4.0f*3.1415926536f));
  atomicAdd(&tw[i], locw / (4.0f*3.1415926536f));

  return;
}

// -------------------------
// update kernel - GPU
__global__ void nbody_3d_posupdate_gpu(
    const FLOAT dt,
    const int32_t tOffset,
    FLOAT* const __restrict__ tx,
    FLOAT* const __restrict__ ty,
    FLOAT* const __restrict__ tz,
    const FLOAT* const __restrict__ tu,
    const FLOAT* const __restrict__ tv,
    const FLOAT* const __restrict__ tw) {

  // local "thread" id - this is the target particle
  const int32_t i = tOffset + blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;

  // save into device view - no need for atomics
  tx[i] += dt * tu[i];
  ty[i] += dt * tv[i];
  tz[i] += dt * tw[i];

  return;
}

// -------------------------
// compute kernel - CPU
__host__ void ngrav_3d_nograds_cpu(
    const int32_t nSrc,
    const FLOAT* const __restrict__ sx,
    const FLOAT* const __restrict__ sy,
    const FLOAT* const __restrict__ sz,
    const FLOAT* const __restrict__ ss,
    const FLOAT* const __restrict__ sr,
    const int32_t nTrg,
    const FLOAT* const __restrict__ tx,
    const FLOAT* const __restrict__ ty,
    const FLOAT* const __restrict__ tz,
    const FLOAT* const __restrict__ tr,
    FLOAT* const __restrict__ tu,
    FLOAT* const __restrict__ tv,
    FLOAT* const __restrict__ tw) {

  // velocity accumulators for target point
  FLOAT totu[CPU_TRG_BLK];
  FLOAT totv[CPU_TRG_BLK];
  FLOAT totw[CPU_TRG_BLK];
  for (int32_t i=0; i<nTrg; ++i) {
    totu[i] = 0.0f;
    totv[i] = 0.0f;
    totw[i] = 0.0f;
  }

  assert(nTrg <= CPU_TRG_BLK && "Cpu target block too large");

  // loop over all source points, two tiers of blocks
  // this is only for improved precision
  for (int32_t jbk=0; jbk<((nSrc+CPU_SRC_BLK-1)/CPU_SRC_BLK); ++jbk) {
    const int32_t jstart = CPU_SRC_BLK*jbk;
    const int32_t jend = std::min(nSrc, CPU_SRC_BLK*(jbk+1));

    // loop over the 16-ish target points
    for (int32_t i=0; i<nTrg; ++i) {
      FLOAT locu = 0.0f;
      FLOAT locv = 0.0f;
      FLOAT locw = 0.0f;
      const FLOAT tr2 = tr[i]*tr[i];

      #pragma omp simd reduction(+:locu,locv,locw)
      for (int32_t j=jstart; j<jend; ++j) {
        const FLOAT dx = sx[j] - tx[i];
        const FLOAT dy = sy[j] - ty[i];
        const FLOAT dz = sz[j] - tz[i];
        const FLOAT distsq = dx*dx + dy*dy + dz*dz + sr[j]*sr[j] + tr2;
        const FLOAT factor = ss[j] / (distsq * std::sqrt(distsq));
        locu += dx * factor;
        locv += dy * factor;
        locw += dz * factor;
      }

      totu[i] += locu;
      totv[i] += locv;
      totw[i] += locw;
    }
  }

  // save into main array
  for (int32_t i=0; i<nTrg; ++i) {
    tu[i] = totu[i] / (4.0f*3.1415926536f);
    tv[i] = totv[i] / (4.0f*3.1415926536f);
    tw[i] = totw[i] / (4.0f*3.1415926536f);
  }

  return;
}

// not really alignment, just minimum block sizes
__host__ int32_t buffer(const int32_t _n, const int32_t _align) {
  // 63,64 returns 1; 64,64 returns 1; 65,64 returns 2
  return _align*(1+(_n-1)/_align);
}

// main program

static void usage() {
  fprintf(stderr, "Usage: ngHip05.bin [-n=<num parts>] [-g=<num gpus>] [-s=<num steps>]\n");
  exit(1);
}

int main(int argc, char **argv) {

  // number of particles/points, gpus, time steps
  int32_t npart = 400000;
  int32_t force_ngpus = -1;
  int32_t nsteps = 1;

  for (int i=1; i<argc; i++) {
    if (strncmp(argv[i], "-n=", 3) == 0) {
      int32_t num = atoi(argv[i]+3);
      if (num < 1) usage();
      npart = num;
    } else if (strncmp(argv[i], "-g=", 3) == 0) {
      int32_t num = atof(argv[i]+3);
      if (num < 1 or num > MAX_GPUS) usage();
      force_ngpus = num;
    } else if (strncmp(argv[i], "-s=", 3) == 0) {
      int32_t num = atoi(argv[i]+3);
      if (num < 1) usage();
      nsteps = num;
    }
  }

  printf( "performing 3D gravitational summation on %d points for %d steps\n", npart, nsteps);
  const FLOAT dt = 0.01;

  // number of GPUs present
  int32_t ngpus = 1;
  hipGetDeviceCount(&ngpus);
  if (force_ngpus > 0) ngpus = force_ngpus;
  // number of streams to break work into
  int32_t nstreams = std::min(MAX_GPUS, ngpus);
  printf( "  ngpus ( %d )  and nstreams ( %d )\n", ngpus, nstreams);

  // we parallelize targets over GPUs/streams
  const int32_t ntargperstrm = buffer(npart/nstreams, THREADS_PER_BLOCK*nstreams);
  const int32_t ntargpad = ntargperstrm * nstreams;
  printf( "  ntargperstrm ( %d )  and ntargpad ( %d )\n", ntargperstrm, ntargpad);

  // and on each GPU, we parallelize over THREADS_PER_BLOCK targets and nsrcblocks source blocks
  // number of blocks source-wise (break summations over sources into this many chunks)
  const int32_t nsrcblocks = 64;

  // set stream sizes
  const int32_t nsrcpad = buffer(npart, THREADS_PER_BLOCK*nsrcblocks);
  const int32_t nsrcperblock = nsrcpad / nsrcblocks;
  printf( "  nsrcperblock ( %d )  and nsrcpad ( %d )\n", nsrcperblock, nsrcpad);

  // define the host arrays (for now, sources and targets are the same)
  const int32_t npad = std::max(ntargpad,nsrcpad);
  std::vector<FLOAT> hsx(npad), hsy(npad), hsz(npad), hss(npad), hsr(npad), htu(npad), htv(npad), htw(npad);
  const FLOAT thisstrmag = 1.0 / std::sqrt(npart);
  const FLOAT thisrad    = (2./3.) / std::sqrt(npart);
  //std::random_device dev;
  //std::mt19937 rng(dev());
  std::mt19937 rng(1234);
  std::uniform_real_distribution<FLOAT> xrand(0.0,1.0);
  for (int32_t i = 0; i < npart; ++i)    hsx[i] = xrand(rng);
  for (int32_t i = npart; i < npad; ++i) hsx[i] = 0.0;
  for (int32_t i = 0; i < npart; ++i)    hsy[i] = xrand(rng);
  for (int32_t i = npart; i < npad; ++i) hsy[i] = 0.0;
  for (int32_t i = 0; i < npart; ++i)    hsz[i] = xrand(rng);
  for (int32_t i = npart; i < npad; ++i) hsz[i] = 0.0;
  for (int32_t i = 0; i < npart; ++i)    hss[i] = thisstrmag * xrand(rng);
  for (int32_t i = npart; i < npad; ++i) hss[i] = 0.0;
  for (int32_t i = 0; i < npart; ++i)    hsr[i] = thisrad;
  for (int32_t i = npart; i < npad; ++i) hsr[i] = thisrad;

  // -------------------------
  // do a CPU version

  auto start = std::chrono::system_clock::now();

  for (int32_t istep=0; istep<nsteps; ++istep) {

    // zero accumulation buffers
    for (int32_t i = 0; i < npad; ++i) htu[i] = 0.0;
    for (int32_t i = 0; i < npad; ++i) htv[i] = 0.0;
    for (int32_t i = 0; i < npad; ++i) htw[i] = 0.0;

    // acceleration-finding kernel
    #pragma omp parallel for schedule(guided)
    for (int32_t ibk=0; ibk<((npart+CPU_TRG_BLK-1)/CPU_TRG_BLK); ++ibk) {
      const int32_t istart = CPU_TRG_BLK*ibk;
      const int32_t iend = std::min(npart, CPU_TRG_BLK*(ibk+1));
      ngrav_3d_nograds_cpu(npart, hsx.data(),hsy.data(),hsz.data(),hss.data(),hsr.data(),
                           iend-istart, &hsx[istart],&hsy[istart],&hsz[istart],&hsr[istart],
                           &htu[istart],&htv[istart],&htw[istart]);
    }

    // position update (simple euler step)
    #pragma omp parallel for schedule(guided)
    for (int32_t i=0; i<npart; ++i) {
      hsx[i] += dt * htu[i];
      hsy[i] += dt * htv[i];
      hsz[i] += dt * htw[i];
    }

  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  double time = elapsed_seconds.count();

  printf( "  host total time( %g s ) and flops( %g GFlop/s )\n", time, nsteps*1.e-9 * (double)npart*(7+20*(double)npart)/time);
  printf( "    results ( %g %g %g %g %g %g)\n", htu[0], htv[0], htw[0], htu[npart-1], htv[npart-1], htw[npart-1]);

  // copy the results into temp vectors
  std::vector<FLOAT> htu_cpu(htu);
  std::vector<FLOAT> htv_cpu(htv);
  std::vector<FLOAT> htw_cpu(htw);

  // -------------------------
  // do the GPU version

  // set device pointers, too
  FLOAT *dsx[MAX_GPUS], *dsy[MAX_GPUS], *dsz[MAX_GPUS], *dss[MAX_GPUS], *dsr[MAX_GPUS];
  FLOAT *dtx[MAX_GPUS], *dty[MAX_GPUS], *dtz[MAX_GPUS], *dtr[MAX_GPUS];
  FLOAT *dtu[MAX_GPUS], *dtv[MAX_GPUS], *dtw[MAX_GPUS];
  hipStream_t stream[MAX_GPUS];

  // allocate space for all sources, part of targets
  const int32_t srcsize = nsrcpad*sizeof(FLOAT);
  const int32_t trgsize = ntargperstrm*sizeof(FLOAT);
  for (int32_t i=0; i<nstreams; ++i) {
    hipSetDevice(i);
    hipStreamCreate(&stream[i]);

    hipMalloc (&dsx[i], srcsize);
    hipMalloc (&dsy[i], srcsize);
    hipMalloc (&dsz[i], srcsize);
    hipMalloc (&dss[i], srcsize);
    hipMalloc (&dsr[i], srcsize);
    hipMalloc (&dtu[i], trgsize);
    hipMalloc (&dtv[i], trgsize);
    hipMalloc (&dtw[i], trgsize);
  }

  const dim3 blocksz(THREADS_PER_BLOCK, 1, 1);
  const dim3 gridsz(ntargperstrm/THREADS_PER_BLOCK, nsrcblocks, 1);
  const dim3 gridupdate(ntargperstrm/THREADS_PER_BLOCK, 1, 1);

  // to be fair, we start timer after allocation but before transfer
  start = std::chrono::system_clock::now();

  // now perform the data movement and setting
  for (int32_t i=0; i<nstreams; ++i) {

    hipSetDevice(i);

    // set some and move other data
    hipMemcpyAsync (dsx[i], hsx.data(), srcsize, hipMemcpyHostToDevice, stream[i]);
    hipMemcpyAsync (dsy[i], hsy.data(), srcsize, hipMemcpyHostToDevice, stream[i]);
    hipMemcpyAsync (dsz[i], hsz.data(), srcsize, hipMemcpyHostToDevice, stream[i]);
    hipMemcpyAsync (dss[i], hss.data(), srcsize, hipMemcpyHostToDevice, stream[i]);
    hipMemcpyAsync (dsr[i], hsr.data(), srcsize, hipMemcpyHostToDevice, stream[i]);
    // now we need to be careful to point to the part of the source arrays that hold
    //   just this GPUs set of target particles
    dtx[i] = dsx[i] + i*ntargperstrm;
    dty[i] = dsy[i] + i*ntargperstrm;
    dtz[i] = dsz[i] + i*ntargperstrm;
    dtr[i] = dsr[i] + i*ntargperstrm;

    // check
    //auto memerr = hipGetLastError();
    //if (memerr != hipSuccess) {
    //  fprintf(stderr, "Failed to upload data (other): %s!\n", hipGetErrorString(memerr));
    //  exit(EXIT_FAILURE);
    //}
  }

  for (int32_t istep=0; istep<nsteps; ++istep) {

  for (int32_t i=0; i<nstreams; ++i) {
    // get this device and stream
    hipSetDevice(i);

    // zero the accumulators
    hipMemsetAsync (dtu[i], 0, trgsize, stream[i]);
    hipMemsetAsync (dtv[i], 0, trgsize, stream[i]);
    hipMemsetAsync (dtw[i], 0, trgsize, stream[i]);

    // launch the kernels
    hipLaunchKernelGGL(ngrav_3d_nograds_gpu, dim3(gridsz), dim3(blocksz), 0, stream[i],
                       nsrcpad, dsx[i],dsy[i],dsz[i],dss[i],dsr[i],
                       0,dtx[i],dty[i],dtz[i],dtr[i],dtu[i],dtv[i],dtw[i]);

    hipLaunchKernelGGL(nbody_3d_posupdate_gpu, dim3(gridupdate), dim3(blocksz), 0, stream[i],
                       dt,0,dtx[i],dty[i],dtz[i],dtu[i],dtv[i],dtw[i]);

    // check
    //auto err = hipGetLastError();
    //if (err != hipSuccess) {
    //  fprintf(stderr, "Failed to launch kernel (%d): %s!\n", i, hipGetErrorString(err));
    //  exit(EXIT_FAILURE);
    //}

    // copy what was changed to all other GPUs
    for (int32_t dstDev=0; dstDev<ngpus; ++dstDev) {
      if (dstDev != i) {
        hipMemcpyPeerAsync (dsx[dstDev], dstDev, dtx.data(), i, trgsize, stream[i]);
      }
    }

    // synchronize so that position arrays are consistent
    hipDeviceSynchronize();
  }

  }

  // moving these calls inside of the kernel loop slows things down a lot
  for (int32_t i=0; i<nstreams; ++i) {
    // pull data back down
    hipMemcpyAsync (htu.data() + i*ntargperstrm, dtu[i], trgsize, hipMemcpyDeviceToHost, stream[i]);
    hipMemcpyAsync (htv.data() + i*ntargperstrm, dtv[i], trgsize, hipMemcpyDeviceToHost, stream[i]);
    hipMemcpyAsync (htw.data() + i*ntargperstrm, dtw[i], trgsize, hipMemcpyDeviceToHost, stream[i]);
  }

  // join streams
  for (int32_t i=0; i<nstreams; ++i) {
    hipStreamSynchronize(stream[i]);
  }

  // time and report
  end = std::chrono::system_clock::now();
  elapsed_seconds = end-start;
  time = elapsed_seconds.count();
  printf( "  device total time( %g s ) and flops( %g GFlop/s )\n", time, 1.e-9 * (double)npart*(7+20*(double)npart)/time);
  printf( "    results ( %g %g %g %g %g %g)\n", htu[0], htv[0], htw[0], htu[npart-1], htv[npart-1], htw[npart-1]);

  // free resources, after timer
  for (int32_t i=0; i<nstreams; ++i) {
    hipFree(dsx[i]);
    hipFree(dsy[i]);
    hipFree(dsz[i]);
    hipFree(dss[i]);
    hipFree(dsr[i]);
    hipFree(dtu[i]);
    hipFree(dtv[i]);
    hipFree(dtw[i]);
    hipStreamDestroy(stream[i]);
  }

  // compare results
  FLOAT errsum = 0.0;
  FLOAT errmax = 0.0;
  for (int32_t i=0; i<npart; ++i) {
    const FLOAT thiserr = std::pow(htu[i]-htu_cpu[i], 2)
                        + std::pow(htv[i]-htv_cpu[i], 2)
                        + std::pow(htw[i]-htw_cpu[i], 2);
    errsum += thiserr;
    if ((FLOAT)std::sqrt(thiserr) > errmax) {
      errmax = (FLOAT)std::sqrt(thiserr);
      //printf( "    err at %d is %g\n", i, errmax);
    }
  }
  printf( "  total host-device error ( %g ) max error ( %g )\n", std::sqrt(errsum/npart), errmax);
}

