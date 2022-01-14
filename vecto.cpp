#include<iostream>
#include<thread>
#include<chrono>
#include<vector>
#include<atomic>
#include<cstdlib>
#include<cassert>
#include<cmath>

#if __ARM_NEON
#include <arm_neon.h>
#define ARM_SIMD
#endif


#define ADD
#define MULTTHREAD
//#define MULT
//#define MPI_TEST_ALY
#define OMP_TEST_ALY

#ifdef OMP_TEST_ALY
#include<omp.h>
#endif
// <
inc

#define N 100000000
#define M 1
#define num_thread 8
#define For(i,N,step) for(i = 0; i < N; i +=step)
#define ForD(i,debut,fin,step) for(i = debut; i < fin; i +=step)
#define IndMat(i,j,m) [i*m + j]

// volatile float Tsum[num_thread];

/*
 *    spin lock
 */
// std::atomic<int> slock(0);

// void spin_lock() {
//   while(std::atomic_exchange(&slock, 1)) { _mm_pause(); }
// }

// void spin_unlock() {
//   std::atomic_store(&slock, 0);
// }

#ifdef ARM_SIMD
void addsimd(uint j,float * __restrict__ u, float * __restrict__ v,float * __restrict__ c)
{
  uint step =16;
  uint n = N/num_thread;
  uint debut = j *n;
  uint fin = (j+1)*n;
  uint q =fin/step;
  uint i = 0;
  
  //#pragma vector always
  //spin_lock();
  ForD(i,debut,q,step)
    {
      // 0 a 8, premiere vague
      float32x4_t armu1 = vld1q_f32(u+i);
      float32x4_t armu2 = vld1q_f32(u+i+4);
      float32x4_t armv1 = vld1q_f32(v+i);
      float32x4_t armv2 = vld1q_f32(v+i+4);
      float32x4_t armc1 = vaddq_f32(armu1,armv1);
      float32x4_t armc2 = vaddq_f32(armu2,armv2);

      // 8 a 16 deuxiemme vague
      float32x4_t armu11 = vld1q_f32(u+i+8);
      float32x4_t armu22 = vld1q_f32(u+i+12);
      float32x4_t armv11 = vld1q_f32(v+i+8);
      float32x4_t armv22 = vld1q_f32(v+i+12);
      float32x4_t armc11 = vaddq_f32(armu11,armv11);
      float32x4_t armc22 = vaddq_f32(armu22,armv22);

      // stockage
      vst1q_f32(c+i,armc1);
      vst1q_f32(c+i+4,armc2);
      vst1q_f32(c+i+8,armc11);
      vst1q_f32(c+i+12,armc22);
    }

  uint r =q*step;
  #pragma ivdep
  ForD(i,r,fin,1) c[i] =u[i]+v[i];

  
}

#endif

#ifdef OMP_TEST_ALY

float * addomp(float * __restrict__ u, float * __restrict__ v, float * __restrict__ c)
{
    uint i = 0;
    // #pragma ivdep
    //#pragma vector always
#pragma omp parallel
    {
      #pragma omp for simd
      For(i,N,1)
	{
	  
	  c[i] = u[i] + v[i];
	}
    }
    
    return c;
}

#endif


float * add(float * __restrict__ u, float * __restrict__ v, float * __restrict__ c)
{
    uint i = 0;
    // #pragma ivdep
    //#pragma vector always
    For(i,N,1)
    {
        c[i] = u[i] + v[i];
      
    }
    return c;
}

#ifdef MULTTHREAD
void add2(uint j,float * __restrict__ u, float * __restrict__ v,float * __restrict__ c)
{
    uint n = N/num_thread;
    uint debut = j *n;
    uint fin = (j+1)*n;
    uint i = 0;

    //#pragma vector always
    //spin_lock();
    // #pragma unroll 16
    ForD(i,debut,fin,1)
    {
      c[i] = u[i] + v[i];
    }
    //spin_unlock();
}
#endif

void init(float * x,uint n)
{
  uint i = 0;
  srand(time(0));
  For(i,n,1)
    {
      x[i] = (float)rand()/(float) RAND_MAX;
    }
}




int main(int argc, char * argv[])
{
  uint i = 0, j = 0;
  #ifdef ADD
  float * u,*v;
  float* c;
    //    printf("size=%d",sizeof(float));
    int size_bytes=ceil((float) N/64.)*64*M*sizeof(float);
    //u = new float[N]; v = new float[N]; c = new float[N];
    uint ret = posix_memalign((void**)&u, 64,size_bytes);
    assert(!ret);
    ret = posix_memalign((void**)&v, 64, N*M*sizeof(float));
    assert(!ret);
    ret = posix_memalign((void**)&c, 64, N*M*sizeof(float));
    assert(!ret);
    init(u,N); init(v,N);
    // Partie Calcul
#ifdef ARM_SIMD
    { 
      auto startt = std::chrono::steady_clock::now();
      std::vector<std::thread> Th;
      i = 0;
      For(i,num_thread,1)
	{
	  Th.push_back(std::thread(addsimd,i,u,v,c));
	}

      For(i,num_thread,1){
        Th[i].join();
      }
      auto endd = std::chrono::steady_clock::now();
      std::chrono::duration<float> elapsed_secondss = endd -startt;
      std::cout<<"le temps ecoule thread simd :"<<elapsed_secondss.count()<<"\n";
      
    }
    // 2
     { 
      auto startt = std::chrono::steady_clock::now();
      std::vector<std::thread> Th;
      i = 0;
      For(i,num_thread,1)
	{
	  Th.push_back(std::thread(addsimd,i,u,v,c));
	}

      For(i,num_thread,1){
        Th[i].join();
      }
      auto endd = std::chrono::steady_clock::now();
      std::chrono::duration<float> elapsed_secondss = endd -startt;
      std::cout<<"le temps ecoule thread simd :"<<elapsed_secondss.count()<<"\n";
      
    }
     // 3
      { 
      auto startt = std::chrono::steady_clock::now();
      std::vector<std::thread> Th;
      i = 0;
      For(i,num_thread,1)
	{
	  Th.push_back(std::thread(addsimd,i,u,v,c));
	}

      For(i,num_thread,1){
        Th[i].join();
      }
      auto endd = std::chrono::steady_clock::now();
      std::chrono::duration<float> elapsed_secondss = endd -startt;
      std::cout<<"le temps ecoule thread simd :"<<elapsed_secondss.count()<<"\n";
      
    }
      // 3
       { 
      auto startt = std::chrono::steady_clock::now();
      std::vector<std::thread> Th;
      i = 0;
      For(i,num_thread,1)
	{
	  Th.push_back(std::thread(addsimd,i,u,v,c));
	}

      For(i,num_thread,1){
        Th[i].join();
      }
      auto endd = std::chrono::steady_clock::now();
      std::chrono::duration<float> elapsed_secondss = endd -startt;
      std::cout<<"le temps ecoule thread simd :"<<elapsed_secondss.count()<<"\n";
      
    }
       // 4
        { 
      auto startt = std::chrono::steady_clock::now();
      std::vector<std::thread> Th;
      i = 0;
      For(i,num_thread,1)
	{
	  Th.push_back(std::thread(addsimd,i,u,v,c));
	}

      For(i,num_thread,1){
        Th[i].join();
      }
      auto endd = std::chrono::steady_clock::now();
      std::chrono::duration<float> elapsed_secondss = endd -startt;
      std::cout<<"le temps ecoule thread simd:"<<elapsed_secondss.count()<<"\n";
      
    }
#endif
// Partie 1
    for(int i=0;i<4;i++) 
    {
      auto start = std::chrono::steady_clock::now();
      c = add(u,v,c);
      auto end = std::chrono::steady_clock::now();
      std::chrono::duration<float> elapsed_seconds = end -start;

      std::cout<<"le temps ecoule1 :"<<elapsed_seconds.count()<<"\n";
    }

    

    // partie 2
 #ifdef MULTTHREAD 
    { 
      auto startt = std::chrono::steady_clock::now();
      std::vector<std::thread> Th;
      i = 0;
      For(i,num_thread,1)
	{
	  Th.push_back(std::thread(add2,i,u,v,c));
	}

      For(i,num_thread,1){
        Th[i].join();
      }
      auto endd = std::chrono::steady_clock::now();
      std::chrono::duration<float> elapsed_secondss = endd -startt;
      std::cout<<"le temps ecoule thread :"<<elapsed_secondss.count()<<"\n";
      
    }
    // 2
     { 
      auto startt = std::chrono::steady_clock::now();
      std::vector<std::thread> Th;
      i = 0;
      For(i,num_thread,1)
	{
	  Th.push_back(std::thread(add2,i,u,v,c));
	}

      For(i,num_thread,1){
        Th[i].join();
      }
      auto endd = std::chrono::steady_clock::now();
      std::chrono::duration<float> elapsed_secondss = endd -startt;
      std::cout<<"le temps ecoule thread :"<<elapsed_secondss.count()<<"\n";
      
    }
     // 3
      { 
      auto startt = std::chrono::steady_clock::now();
      std::vector<std::thread> Th;
      i = 0;
      For(i,num_thread,1)
	{
	  Th.push_back(std::thread(add2,i,u,v,c));
	}

      For(i,num_thread,1){
        Th[i].join();
      }
      auto endd = std::chrono::steady_clock::now();
      std::chrono::duration<float> elapsed_secondss = endd -startt;
      std::cout<<"le temps ecoule thread :"<<elapsed_secondss.count()<<"\n";
      
    }
      // 3
       { 
      auto startt = std::chrono::steady_clock::now();
      std::vector<std::thread> Th;
      i = 0;
      For(i,num_thread,1)
	{
	  Th.push_back(std::thread(add2,i,u,v,c));
	}

      For(i,num_thread,1){
        Th[i].join();
      }
      auto endd = std::chrono::steady_clock::now();
      std::chrono::duration<float> elapsed_secondss = endd -startt;
      std::cout<<"le temps ecoule thread :"<<elapsed_secondss.count()<<"\n";
      
    }
       // 4
        { 
      auto startt = std::chrono::steady_clock::now();
      std::vector<std::thread> Th;
      i = 0;
      For(i,num_thread,1)
	{
	  Th.push_back(std::thread(add2,i,u,v,c));
	}

      For(i,num_thread,1){
        Th[i].join();
      }
      auto endd = std::chrono::steady_clock::now();
      std::chrono::duration<float> elapsed_secondss = endd -startt;
      std::cout<<"le temps ecoule thread :"<<elapsed_secondss.count()<<"\n";
      
    }
      
#endif

#ifdef OMP_TEST_ALY
    for(int i=0;i<4;i++) 
    {
      //omp_set_num_threads(num_thread);
      auto start = std::chrono::steady_clock::now();
      c = addomp(u,v,c);
      auto end = std::chrono::steady_clock::now();
      std::chrono::duration<float> elapsed_seconds = end -start;

      std::cout<<"le temps ecoule omp :"<<elapsed_seconds.count()<<"\n";
    }
#endif

    

    // delete[] u;
    // delete[] v;
    // delete[] c;
    free(u);
    free(v); free(c);
#endif


    #ifdef MPI_TEST_ALY
    // --------------------Partie mpi------------------------------//

    {
      auto start4 = std::chrono::steady_clock::now();

      int rank,nbproc;
      // On Commence du mpi ici
      MPI_Init(&argc,&argv);
      MPI_Comm_size(MPI_COMM_WORLD, &nbproc);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      int const taille_par_proc = N*M/nbproc;
      assert(! (N*M % nbproc) );
      float * sous_matrice = nullptr; float * Sous_Y=nullptr;

      int r = posix_memalign((void**)&sous_matrice, 64, taille_par_proc*sizeof(float));
      assert(!r);

      r = posix_memalign((void**)&Sous_Y, 64, taille_par_proc*sizeof(float));
      assert(!r);

      // Envoie des sous-matrices a chaque proc
      MPI_Scatter(A,taille_par_proc,MPI_FLOAT,sous_matrice,taille_par_proc,MPI_FLOAT,0,MPI_COMM_WORLD);

      MPI_Bcast(X,N,MPI_FLOAT,0,MPI_COMM_WORLD); // Second membre

      // On attend que tout le monde reçoive
      MPI_Barrier(MPI_COMM_WORLD);

      // Tache a faire
      Sous_Y = ProdMatvect(A,X,Sous_Y,taille_par_proc,M);

      // On recupere les differents morceaux
      MPI_Gather(Sous_Y,taille_par_proc,MPI_FLOAT,Y,taille_par_proc,MPI_FLOAT,0,MPI_COMM_WORLD);

      // On Attend la fin
      MPI_Barrier(MPI_COMM_WORLD);

      free(Sous_Y); free(sous_matrice);
      MPI_Finalize();
      // Fin du mpi

      auto end4 = std::chrono::steady_clock::now();
      std::chrono::duration<float> elapsed_seconds4 = end4 -start4;
      std::cout<<"le temps Mult_MPI :"<<elapsed_seconds4.count()<<"\n";

    }

    free(A);free(X);free(Y);
#endif



    return 0;
}
