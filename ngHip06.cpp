/*
 * ngHip06.hip
 *
 * (c)2022 Mark J. Stock <markjstock@gmail.com>
 *
 * v0.6  using launch_bounds, other stuff
 */

#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <sched.h>
#include <omp.h>

#include <hip/hip_runtime.h>


// compute using float or double
#define FLOAT float
#define RSQRT rsqrtf
//#define FLOAT double
//#define RSQRT rsqrt

#define CPU_SRC_BLK 256
#define CPU_TRG_BLK 32

// threads per block (hard coded)
#define THREADS_PER_BLOCK 256

// GPU count limit
#define MAX_GPUS 8

// useful macros
#define gpuCheckCall(call)	\
do {							\
  hipError_t err = call;		\
  if (err != hipSuccess) {		\
    fprintf(stderr, "GPU error %s:%d: '%s'!\n", __FILE__, __LINE__, hipGetErrorString(err));	\
    exit(EXIT_FAILURE);			\
  }								\
} while(0)


// -------------------------
// compute kernel - GPU
__global__ void __launch_bounds__(THREADS_PER_BLOCK) ngrav_3d_nograds_gpu(
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

  const FLOAT tr2 = tr[i]*tr[i];

  // which sources do we iterate over?
  const int32_t jcount = nSrc / gridDim.y;
  const int32_t jstart = blockIdx.y * jcount;

  for (int32_t b=0; b<jcount/THREADS_PER_BLOCK; ++b) {

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

    __syncthreads();
  }

  // save into device view with atomics
  atomicAdd(&tu[i], locu / (4.0f*3.1415926536f));
  atomicAdd(&tv[i], locv / (4.0f*3.1415926536f));
  atomicAdd(&tw[i], locw / (4.0f*3.1415926536f));

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
  return _align*((_n+_align-1)/_align);
}

// main program

static void usage() {
  fprintf(stderr, "Usage: ngHip06.bin [-n=<num parts>] [-g=<num gpus>] [-c]\n");
  exit(1);
}

int main(int argc, char **argv) {

  // number of particles/points and gpus
  int32_t npart = 400000;
  int32_t force_ngpus = -1;
  bool compare = false;

  for (int i=1; i<argc; i++) {
    if (strncmp(argv[i], "-n=", 3) == 0) {
      int32_t num = atoi(argv[i]+3);
      if (num < 1) usage();
      npart = num;
    } else if (strncmp(argv[i], "-g=", 3) == 0) {
      int32_t num = atof(argv[i]+3);
      if (num < 1 or num > MAX_GPUS) usage();
      force_ngpus = num;
    } else if (strncmp(argv[i], "-c", 2) == 0) {
      compare = true;
    }
  }

  printf( "performing 3D gravitational summation on %d points\n", npart);

  // number of GPUs present
  int32_t ngpus = 1;
  hipGetDeviceCount(&ngpus);
  if (force_ngpus > 0) ngpus = force_ngpus;
  // number of streams to break work into
  int32_t nstreams = std::min(MAX_GPUS, ngpus);
  printf( "  ngpus ( %d )  and nstreams ( %d )\n", ngpus, nstreams);

  // we parallelize targets over GPUs/streams
  const int32_t ntargpad = buffer(npart, THREADS_PER_BLOCK*nstreams);
  const int32_t ntargperstrm = ntargpad / nstreams;
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
  {
  auto start = std::chrono::system_clock::now();
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
  for (int32_t i = 0; i < npad; ++i)     htu[i] = 0.0;
  for (int32_t i = 0; i < npad; ++i)     htv[i] = 0.0;
  for (int32_t i = 0; i < npad; ++i)     htw[i] = 0.0;
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  double time = elapsed_seconds.count();
  printf( "  host alloc time( %g s )\n", time);
  }

  // -------------------------
  // do a CPU version

  if (compare) {
  auto start = std::chrono::system_clock::now();

  #pragma omp parallel for schedule(guided)
  for (int32_t ibk=0; ibk<((npart+CPU_TRG_BLK-1)/CPU_TRG_BLK); ++ibk) {
    const int32_t istart = CPU_TRG_BLK*ibk;
    const int32_t iend = std::min(npart, CPU_TRG_BLK*(ibk+1));
    ngrav_3d_nograds_cpu(npart, hsx.data(),hsy.data(),hsz.data(),hss.data(),hsr.data(),
                         iend-istart, &hsx[istart],&hsy[istart],&hsz[istart],&hsr[istart],
                         &htu[istart],&htv[istart],&htw[istart]);
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  double time = elapsed_seconds.count();

  printf( "  host compute time( %g s ) and flops( %g GFlop/s )\n", time, 1.e-9 * (double)npart*(7+20*(double)npart)/time);
  printf( "    results ( %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f )\n", htu[0], htv[0], htw[0], htu[npart-1], htv[npart-1], htw[npart-1]);
  }

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
  const int32_t srcsize = nsrcpad*sizeof(FLOAT);
  const int32_t trgsize = ntargperstrm*sizeof(FLOAT);
  const std::vector<int32_t> gpubind = {4,5,2,3,6,7,0,1};

  {
  auto start = std::chrono::system_clock::now();
  // allocate space for all sources, part of targets
  #pragma omp parallel for schedule(static,1) num_threads(nstreams)
  for (int32_t i=0; i<nstreams; ++i) {

    // linear device setting, openmp thread = gpuid
    //const int gpuid = i;
    // account for nearest GPU
    const int gpuid = gpubind[sched_getcpu()/8];
    hipSetDevice(gpuid);

    auto scstart = std::chrono::system_clock::now();
    //std::cout << "  omp thread " << omp_get_thread_num() << " on cpu " << sched_getcpu() << " creating stream " << i << std::endl;

    // standard launch, synchronous? allocates on heap
    hipStreamCreate(&stream[i]);
    // with non-blocking flag - very little difference under openmp
    //hipStreamCreateWithFlags(&stream[i], hipStreamNonBlocking);

    auto scend = std::chrono::system_clock::now();
    std::chrono::duration<double> scelapsed_seconds = scend-scstart;
    std::cout << "  omp thread " << omp_get_thread_num() << " on cpu " << sched_getcpu() << " finished stream " << i << " on gpu " << gpuid << " in " << scelapsed_seconds.count() << std::endl;

    hipMalloc (&dsx[i], srcsize);
    hipMalloc (&dsy[i], srcsize);
    hipMalloc (&dsz[i], srcsize);
    hipMalloc (&dss[i], srcsize);
    hipMalloc (&dsr[i], srcsize);
    hipMalloc (&dtu[i], trgsize);
    hipMalloc (&dtv[i], trgsize);
    hipMalloc (&dtw[i], trgsize);
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  double time = elapsed_seconds.count();
  printf( "  device alloc time( %g s )\n", time);
  }

  const dim3 blocksz(THREADS_PER_BLOCK, 1, 1);
  const dim3 gridsz(ntargperstrm/THREADS_PER_BLOCK, nsrcblocks, 1);

  // to be fair, we start timer after allocation but before transfer
  auto start = std::chrono::system_clock::now();

  // now perform the data movement and setting
  #pragma omp parallel for schedule(static,1) num_threads(nstreams)
  for (int32_t i=0; i<nstreams; ++i) {

    hipSetDevice(i);

    // set some and move other data
    hipMemsetAsync (dtu[i], 0, trgsize, stream[i]);
    hipMemsetAsync (dtv[i], 0, trgsize, stream[i]);
    hipMemsetAsync (dtw[i], 0, trgsize, stream[i]);
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
    if (false) gpuCheckCall(hipGetLastError());
  //}

  //for (int32_t i=0; i<nstreams; ++i) {
    // get this device and stream
    //hipSetDevice(i);

    // launch the kernels
    hipLaunchKernelGGL(ngrav_3d_nograds_gpu, dim3(gridsz), dim3(blocksz), 0, stream[i],
                       nsrcpad, dsx[i],dsy[i],dsz[i],dss[i],dsr[i],
                       0,dtx[i],dty[i],dtz[i],dtr[i],dtu[i],dtv[i],dtw[i]);

    // check for synchronous errors
    if (false) gpuCheckCall(hipGetLastError());
  }

  // moving these calls inside of the kernel loop slows things down a lot
  #pragma omp parallel for schedule(static,1) num_threads(nstreams)
  for (int32_t i=0; i<nstreams; ++i) {
    // pull data back down
    hipMemcpyAsync (htu.data() + i*ntargperstrm, dtu[i], trgsize, hipMemcpyDeviceToHost, stream[i]);
    hipMemcpyAsync (htv.data() + i*ntargperstrm, dtv[i], trgsize, hipMemcpyDeviceToHost, stream[i]);
    hipMemcpyAsync (htw.data() + i*ntargperstrm, dtw[i], trgsize, hipMemcpyDeviceToHost, stream[i]);
  }

  // join streams
  for (int32_t i=0; i<nstreams; ++i) {
    gpuCheckCall( hipStreamSynchronize(stream[i]) );
  }

  // check for asynchronous errors
  // which device?
  if (false) gpuCheckCall( hipDeviceSynchronize() );

  // time and report
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  double time = elapsed_seconds.count();
  printf( "  device comm+comp time( %g s ) and flops( %g GFlop/s )\n", time, 1.e-9 * (double)npart*(7+20*(double)npart)/time);
  printf( "    results ( %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f )\n", htu[0], htv[0], htw[0], htu[npart-1], htv[npart-1], htw[npart-1]);

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
  if (compare) {
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
}

