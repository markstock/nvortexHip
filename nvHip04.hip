/*
 * nvHip04.hip
 *
 * (c)2022 Mark J. Stock <markjstock@gmail.com>
 *
 * v0.4  use Kahan summation to improve answer
 */

#include <vector>
#include <random>
#include <chrono>

#include <hip/hip_runtime.h>


// compute using float or double
#define FLOAT2 float2
#define FLOAT float

// threads per block (hard coded)
#define THREADS_PER_BLOCK 512

// GPU count limit
#define MAX_GPUS 8

// Kahan summation
// use single precision for the storage and arithmetic, but accumulation acts close to double-precision
// sum.x is running sum, sum.y is compensation/error
__device__ inline void KahanSum_gpu (const FLOAT toadd, FLOAT2* const sum) {
  const FLOAT y = toadd - (*sum).y;
  const FLOAT t = (*sum).x + y;
  //const volatile FLOAT t = (*sum).x + y;
  //const volatile FLOAT z = t - (*sum).x;
  //(*sum).y = z - y;
  (*sum).y = (t - (*sum).x) - y;
  (*sum).x = t;
}

// -------------------------
// compute kernel - GPU
__global__ void nvortex_2d_nograds_gpu(
    const int32_t nSrc,
    const FLOAT* const __restrict__ sx,
    const FLOAT* const __restrict__ sy,
    const FLOAT* const __restrict__ ss,
    const FLOAT* const __restrict__ sr,
    const int32_t tOffset,
    const FLOAT* const __restrict__ tx,
    const FLOAT* const __restrict__ ty,
    const FLOAT* const __restrict__ tr,
    FLOAT* const __restrict__ tu,
    FLOAT* const __restrict__ tv) {

  // local "thread" id - this is the target particle
  const int32_t i = tOffset + blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;

  // load sources into shared memory (or not)
  __shared__ FLOAT s_sx[THREADS_PER_BLOCK];
  __shared__ FLOAT s_sy[THREADS_PER_BLOCK];
  __shared__ FLOAT s_ss[THREADS_PER_BLOCK];
  __shared__ FLOAT s_sr[THREADS_PER_BLOCK];

  // velocity accumulators for target point
  FLOAT2 locu = make_float2(0.f, 0.f);
  FLOAT2 locv = make_float2(0.f, 0.f);

  FLOAT tr2 = tr[i]*tr[i];

  // which sources do we iterate over?
  const int32_t jcount = nSrc / gridDim.y;
  const int32_t jstart = blockIdx.y * jcount;

  for (int32_t b=0; b<jcount/THREADS_PER_BLOCK; ++b) {

    const int32_t gidx = jstart + b*THREADS_PER_BLOCK + threadIdx.x;
    s_sx[threadIdx.x] = sx[gidx];
    s_sy[threadIdx.x] = sy[gidx];
    s_ss[threadIdx.x] = ss[gidx];
    s_sr[threadIdx.x] = sr[gidx];
    __syncthreads();

    // loop over all source points
    for (int32_t j=0; j<THREADS_PER_BLOCK; ++j) {
      FLOAT dx = s_sx[j] - tx[i];
      FLOAT dy = s_sy[j] - ty[i];
      FLOAT distsq = dx*dx + dy*dy + s_sr[j]*s_sr[j] + tr2;
      // we get __fdividef(x, y) with --use_fast_math it seems
      FLOAT factor = s_ss[j] / distsq;
      KahanSum_gpu( dy * factor, &locu);
      KahanSum_gpu(-dx * factor, &locv);
    }

    __syncthreads();
  }

  // save into device view with atomics
  atomicAdd(&tu[i], (locu.x+locu.y) / (2.0f*3.1415926536f));
  atomicAdd(&tv[i], (locv.x+locv.y) / (2.0f*3.1415926536f));

  return;
}

// Kahan summation
// use single precision for the storage and arithmetic, but accumulation acts close to double-precision
// sum.x is running sum, sum.y is compensation/error
#pragma omp declare simd
__host__ inline void KahanSum_cpu (const FLOAT toadd, FLOAT* const sum, FLOAT* const rem) {
  const FLOAT y = toadd - *rem;
  const FLOAT t = *sum + y;
  //const volatile FLOAT t = *sum + y;
  //const volatile FLOAT z = t - *sum;
  //*rem = z - y;
  *rem = (t - *sum) - y;
  *sum = t;
}

// -------------------------
// compute kernel - CPU
__host__ void nvortex_2d_nograds_cpu(
    const int32_t nSrc,
    const FLOAT* const __restrict__ sx,
    const FLOAT* const __restrict__ sy,
    const FLOAT* const __restrict__ ss,
    const FLOAT* const __restrict__ sr,
    const FLOAT tx,
    const FLOAT ty,
    const FLOAT tr,
    FLOAT* const __restrict__ tu,
    FLOAT* const __restrict__ tv) {

  // velocity accumulators for target point
  FLOAT locu = 0.0f;
  FLOAT locv = 0.0f;
  FLOAT ukah = 0.0f;
  FLOAT vkah = 0.0f;

  const FLOAT tr2 = tr*tr;

  // loop over all source points
  #pragma omp simd reduction(+:locu,locv)
  for (int32_t j=0; j<nSrc; ++j) {
    FLOAT dx = sx[j] - tx;
    FLOAT dy = sy[j] - ty;
    FLOAT distsq = dx*dx + dy*dy + sr[j]*sr[j] + tr2;
    FLOAT factor = ss[j] / distsq;
    //{
      FLOAT y = dy * factor - ukah;
      FLOAT t = locu + y;
      ukah = (t - locu) - y;
      locu = t;
    //}
    //{
      y = -dx * factor - vkah;
      t = locv + y;
      vkah = (t - locv) - y;
      locv = t;
    //}
    // I just can't get this to simd-ize!!!
    //KahanSum_cpu( dy * factor, &locu, &ukah);
    //KahanSum_cpu(-dx * factor, &locv, &vkah);
  }

  // save into device view
  // use atomics?!?
  *tu = (locu+ukah) / (2.0f*3.1415926536f);
  *tv = (locv+vkah) / (2.0f*3.1415926536f);

  return;
}

// not really alignment, just minimum block sizes
__host__ int32_t buffer(const int32_t _n, const int32_t _align) {
  // 63,64 returns 1; 64,64 returns 1; 65,64 returns 2
  return _align*(1+(_n-1)/_align);
}

// main program

static void usage() {
  fprintf(stderr, "Usage: nvHip04.bin [-n=<number>]\n");
  exit(1);
}

int main(int argc, char **argv) {

  // number of particles/points
  int32_t npart = 400000;

  if (argc > 1) {
    if (strncmp(argv[1], "-n=", 3) == 0) {
      int num = atoi(argv[1] + 3);
      if (num < 1) usage();
      npart = num;
    }
  }

  printf( "performing 2D vortex Biot-Savart on %d points\n", npart);

  // number of GPUs present
  int32_t ngpus = 1;
  hipGetDeviceCount(&ngpus);
  //ngpus = 1;	// Force 1 GPU
  // number of cuda streams to break work into
  int32_t nstreams = std::min(MAX_GPUS, ngpus);
  printf( "  ngpus ( %d )  and nstreams ( %d )\n", ngpus, nstreams);

  // we parallelize targets over GPUs/streams
  const int32_t ntargpad = buffer(npart, THREADS_PER_BLOCK*nstreams);
  const int32_t ntargperstrm = ntargpad / nstreams;
  printf( "  ntargperstrm ( %d )  and ntargpad ( %d )\n", ntargperstrm, ntargpad);

  // and on each GPU, we parallelize over THREADS_PER_BLOCK targets and nsrcblocks source blocks
  // number of blocks source-wise (break summations over sources into this many chunks)
  const int32_t nsrcblocks = 32;

  // set stream sizes
  const int32_t nsrcpad = buffer(npart, THREADS_PER_BLOCK*nsrcblocks);
  const int32_t nsrcperblock = nsrcpad / nsrcblocks;
  printf( "  nsrcperblock ( %d )  and nsrcpad ( %d )\n", nsrcperblock, nsrcpad);

  // define the host arrays (for now, sources and targets are the same)
  const int32_t npad = std::max(ntargpad,nsrcpad);
  std::vector<FLOAT> hsx(npad), hsy(npad), hss(npad), hsr(npad), htu(npad), htv(npad);
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
  for (int32_t i = 0; i < npart; ++i)    hss[i] = thisstrmag * (2.0*xrand(rng)-1.0);
  for (int32_t i = npart; i < npad; ++i) hss[i] = 0.0;
  for (int32_t i = 0; i < npart; ++i)    hsr[i] = thisrad;
  for (int32_t i = npart; i < npad; ++i) hsr[i] = thisrad;
  for (int32_t i = 0; i < npad; ++i)     htu[i] = 0.0;
  for (int32_t i = 0; i < npad; ++i)     htv[i] = 0.0;

  // -------------------------
  // do a CPU version

  auto start = std::chrono::system_clock::now();

  #pragma omp parallel for schedule(guided)
  for (int32_t i=0; i<npart; ++i) {
    nvortex_2d_nograds_cpu(npart, hsx.data(),hsy.data(),hss.data(),hsr.data(), hsx[i],hsy[i],hsr[i], &htu[i],&htv[i]);
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  double time = elapsed_seconds.count();

  printf( "  host total time( %g s ) and flops( %g GFlop/s )\n", time, 1.e-9 * (double)npart*(7+19*(double)npart)/time);
  printf( "    results ( %g %g %g %g %g %g)\n", htu[0], htv[0], htu[1], htv[1], htu[npart-1], htv[npart-1]);

  // copy the results into temp vectors
  std::vector<FLOAT> htu_cpu(htu);
  std::vector<FLOAT> htv_cpu(htv);

  // -------------------------
  // do the GPU version

  // set device pointers, too
  FLOAT *dsx[MAX_GPUS], *dsy[MAX_GPUS], *dss[MAX_GPUS], *dsr[MAX_GPUS];
  FLOAT *dtx[MAX_GPUS], *dty[MAX_GPUS], *dtr[MAX_GPUS];
  FLOAT *dtu[MAX_GPUS], *dtv[MAX_GPUS];
  hipStream_t stream[MAX_GPUS];

  // allocate space for all sources, part of targets
  const int32_t srcsize = nsrcpad*sizeof(FLOAT);
  const int32_t trgsize = ntargperstrm*sizeof(FLOAT);
  for (int32_t i=0; i<nstreams; ++i) {
    hipSetDevice(i);
    hipStreamCreate(&stream[i]);

    hipMalloc (&dsx[i], srcsize);
    hipMalloc (&dsy[i], srcsize);
    hipMalloc (&dss[i], srcsize);
    hipMalloc (&dsr[i], srcsize);
    hipMalloc (&dtu[i], trgsize);
    hipMalloc (&dtv[i], trgsize);
  }

  const dim3 blocksz(THREADS_PER_BLOCK, 1, 1);
  const dim3 gridsz(ntargperstrm/THREADS_PER_BLOCK, nsrcblocks, 1);

  // to be fair, we start timer after allocation but before transfer
  start = std::chrono::system_clock::now();

  // now perform the data movement and setting
  for (int32_t i=0; i<nstreams; ++i) {

    hipSetDevice(i);

    // move the data
    hipMemcpyAsync (dsx[i], hsx.data(), srcsize, hipMemcpyHostToDevice, stream[i]);
    hipMemcpyAsync (dsy[i], hsy.data(), srcsize, hipMemcpyHostToDevice, stream[i]);
    hipMemcpyAsync (dss[i], hss.data(), srcsize, hipMemcpyHostToDevice, stream[i]);
    hipMemcpyAsync (dsr[i], hsr.data(), srcsize, hipMemcpyHostToDevice, stream[i]);
    hipMemsetAsync (dtu[i], 0, trgsize, stream[i]);
    hipMemsetAsync (dtv[i], 0, trgsize, stream[i]);
    // now we need to be careful to point to the part of the source arrays that hold
    //   just this GPUs set of target particles
    dtx[i] = dsx[i] + i*ntargperstrm;
    dty[i] = dsy[i] + i*ntargperstrm;
    dtr[i] = dsr[i] + i*ntargperstrm;

    // check
    auto memerr = hipGetLastError();
    if (memerr != hipSuccess) {
      fprintf(stderr, "Failed to upload data (other): %s!\n", hipGetErrorString(memerr));
      exit(EXIT_FAILURE);
    }
  }

  for (int32_t i=0; i<nstreams; ++i) {
    // launch the kernel
    hipSetDevice(i);
    hipLaunchKernelGGL(nvortex_2d_nograds_gpu, dim3(gridsz), dim3(blocksz), 0, stream[i], nsrcpad, dsx[i],dsy[i],dss[i],dsr[i],
                                               0,dtx[i],dty[i],dtr[i],dtu[i],dtv[i]);

    // check
    auto err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr, "Failed to launch kernel (%d): %s!\n", i, hipGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

  for (int32_t i=0; i<nstreams; ++i) {
    // pull data back down
    hipMemcpyAsync (htu.data() + i*ntargperstrm, dtu[i], trgsize, hipMemcpyDeviceToHost, stream[i]);
    hipMemcpyAsync (htv.data() + i*ntargperstrm, dtv[i], trgsize, hipMemcpyDeviceToHost, stream[i]);
  }

  // join streams
  for (int32_t i=0; i<nstreams; ++i) {
    hipStreamSynchronize(stream[i]);
  }

  // time and report
  end = std::chrono::system_clock::now();
  elapsed_seconds = end-start;
  time = elapsed_seconds.count();
  printf( "  device total time( %g s ) and flops( %g GFlop/s )\n", time, 1.e-9 * (double)npart*(9+19*(double)npart)/time);
  printf( "    results ( %g %g %g %g %g %g)\n", htu[0], htv[0], htu[1], htv[1], htu[npart-1], htv[npart-1]);

  // free resources, after timer
  for (int32_t i=0; i<nstreams; ++i) {
    hipFree(dsx[i]);
    hipFree(dsy[i]);
    hipFree(dss[i]);
    hipFree(dsr[i]);
    hipFree(dtu[i]);
    hipFree(dtv[i]);
    hipStreamDestroy(stream[i]);
  }

  // compare results
  FLOAT errsum = 0.0;
  FLOAT errmax = 0.0;
  for (int32_t i=0; i<npart; ++i) {
    const FLOAT thiserr = std::pow(htu[i]-htu_cpu[i], 2) + std::pow(htv[i]-htv_cpu[i], 2);
    errsum += thiserr;
    if ((FLOAT)std::sqrt(thiserr) > errmax) {
      errmax = (FLOAT)std::sqrt(thiserr);
      //printf( "    err at %d is %g\n", i, errmax);
    }
  }
  printf( "  total host-device error ( %g ) max error ( %g )\n", std::sqrt(errsum/npart), errmax);
}

