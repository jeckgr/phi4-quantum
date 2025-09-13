#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

/*
 *   A C-program for MT19937-64 (2004/9/29 version).
 *   Coded by Takuji Nishimura and Makoto Matsumoto.
 *
 *   This is a 64-bit version of Mersenne Twister pseudorandom number
 *   generator.
 *
 *   Before using, initialize the state by using init_genrand64(seed)
 *   or init_by_array64(init_key, key_length).
 *
 *   Copyright (C) 2004, Makoto Matsumoto and Takuji Nishimura,
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     1. Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *
 *     2. Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 *     3. The names of its contributors may not be used to endorse or promote
 *        products derived from this software without specific prior written
 *        permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 *   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *   References:
 *   T. Nishimura, ``Tables of 64-bit Mersenne Twisters''
 *     ACM Transactions on Modeling and
 *     Computer Simulation 10. (2000) 348--357.
 *   M. Matsumoto and T. Nishimura,
 *     ``Mersenne Twister: a 623-dimensionally equidistributed
 *       uniform pseudorandom number generator''
 *     ACM Transactions on Modeling and
 *     Computer Simulation 8. (Jan. 1998) 3--30.
 *
 *   Any feedback is very welcome.
 *   http://www.math.hiroshima-u.ac.jp/~m-mat/MT/emt.html
 *   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove spaces)
 */

#define NN 312
#define MM 156
#define MATRIX_A 0xB5026F5AA96619E9ULL
#define UM 0xFFFFFFFF80000000ULL /* Most significant 33 bits */
#define LM 0x7FFFFFFFULL /* Least significant 31 bits */

typedef struct tagMTrand {
    unsigned long long mt[NN];
    int mti;
} MTrand;

/* initializes rand->mt[NN] with a seed */
void init_genrand64(MTrand* rand, unsigned long long seed)
{
    rand->mt[0] = seed;
    for (rand->mti=1; rand->mti<NN; rand->mti++)
        rand->mt[rand->mti] =  (6364136223846793005ULL * (rand->mt[rand->mti-1] ^ (rand->mt[rand->mti-1] >> 62)) + rand->mti);
}

/* generates a random number on [0, 2^64-1]-interval */
unsigned long long genrand64_int64(MTrand* rand)
{
    int i;
    unsigned long long x;
    static unsigned long long mag01[2]={0ULL, MATRIX_A};

    if (rand->mti >= NN) { /* generate NN words at one time */

        /* if init_genrand64() has not been called, */
        /* a default initial seed is used     */
        if (rand->mti == NN+1)
            init_genrand64(rand, 5489ULL);

        for (i=0;i<NN-MM;i++) {
            x = (rand->mt[i]&UM)|(rand->mt[i+1]&LM);
            rand->mt[i] = rand->mt[i+MM] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
        }
        for (;i<NN-1;i++) {
            x = (rand->mt[i]&UM)|(rand->mt[i+1]&LM);
            rand->mt[i] = rand->mt[i+(MM-NN)] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
        }
        x = (rand->mt[NN-1]&UM)|(rand->mt[0]&LM);
        rand->mt[NN-1] = rand->mt[MM-1] ^ (x>>1) ^ mag01[(int)(x&1ULL)];

        rand->mti = 0;
    }

    x = rand->mt[rand->mti++];

    x ^= (x >> 29) & 0x5555555555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    x ^= (x >> 43);

    return x;
}

/* generates a random number on (0,1)-real-interval */
double genrand64_real3(MTrand* rand)
{
    return ((genrand64_int64(rand) >> 12) + 0.5) * (1.0/4503599627370496.0);
}

int main() {
    int ns, nt;
    double m2, epsilon, lambda, tfinal, step, bound;
    FILE *f = fopen("input.txt", "r");
    fscanf(f,"%d\n%d\n%lf\n%lf\n%lf\n%lf\n%lf\n%lf",&ns,&nt,&m2,&epsilon,&lambda,&tfinal,&step,&bound);
    fclose(f);
    int *sup = (int*)malloc(ns*sizeof(int));
    int *sdn = (int*)malloc(ns*sizeof(int));
    int *tup = (int*)malloc(nt*sizeof(int));
    int *tdn = (int*)malloc(nt*sizeof(int));
    double complex *cor = (double complex*)malloc(nt*sizeof(double complex));
    double complex ****phi = (double complex****)malloc(nt*sizeof(double complex***));
    double complex ****psi = (double complex****)malloc(nt*sizeof(double complex***));
    int *seed = (int*)malloc(omp_get_max_threads()*sizeof(int));
    MTrand *rng = (MTrand*)malloc(omp_get_max_threads()*sizeof(MTrand));
    double *kmax_thread = (double*)malloc(omp_get_max_threads()*sizeof(double));
    double gamma = (double)ns/(double)nt;
    double dt = sqrt(gamma);
    for (int i = 0; i < omp_get_max_threads(); ++i) {
        seed[i] = rand();
    }
    for (int t = 0; t < nt; ++t) {
        if (t == 0) {
            tup[t] = t+1;
            tdn[t] = nt-1;
        }
        else if (t == nt-1) {
            tup[t] = 0;
            tdn[t] = t-1;
        }
        else {
            tup[t] = t+1;
            tdn[t] = t-1;
        }
        phi[t] = (double complex***)malloc(ns*sizeof(double complex**));
        psi[t] = (double complex***)malloc(ns*sizeof(double complex**));
        for (int x = 0; x < ns; ++x) {
            if (x == 0) {
                sup[x] = x+1;
                sdn[x] = ns-1;
            }
            else if (x == ns-1) {
                sup[x] = 0;
                sdn[x] = x-1;
            }
            else {
                sup[x] = x+1;
                sdn[x] = x-1;
            }
            phi[t][x] = (double complex**)malloc(ns*sizeof(double complex*));
            psi[t][x] = (double complex**)malloc(ns*sizeof(double complex*));
            for (int y = 0; y < ns; ++y) {
                phi[t][x][y] = (double complex*)malloc(ns*sizeof(double complex));
                psi[t][x][y] = (double complex*)malloc(ns*sizeof(double complex));
            }
        }
    }
    double complex **prop = (double complex**)malloc(nt*sizeof(double complex*));
    double complex **invprop = (double complex**)malloc(nt*sizeof(double complex*));
    for (int t = 0; t < nt; ++t) {
        prop[t] = (double complex*)malloc(2*sizeof(double complex));
        invprop[t] = (double complex*)malloc(2*sizeof(double complex));
    }
    for (int n = 0; n < 2; ++n) {
        for (int t = 0; t < nt; ++t) {
            cor[t] = 0.0 + 0.0 * I;
        }
        for (int i = 0; i < omp_get_max_threads(); ++i) {
            init_genrand64(&rng[i],seed[i]);
        }
        #pragma omp parallel for
        for (int t = 0; t < nt; ++t) {
            for (int x = 0; x < ns; ++x) {
                for (int y = 0; y < ns; ++y) {
                    for (int z = 0; z < ns; ++z) {
                        double u1 = genrand64_real3(&rng[omp_get_thread_num()]);
                        double u2 = genrand64_real3(&rng[omp_get_thread_num()]);
                        double nor = sqrt(-2*log(u1))*cos(2*M_PI*u2);
                        phi[t][x][y][z] = sqrt(2*step)*nor;
                    }
                }
            }
        }
        int count = 0;
        double time = 0;
        while (time < tfinal) {
            for (int i = 0; i < omp_get_max_threads(); ++i) {
                kmax_thread[i] = 0;
            }
            #pragma omp parallel for
            for (int t = 0; t < nt; ++t) {
                for (int x = 0; x < ns; ++x) {
                    for (int y = 0; y < ns; ++y) {
                        for (int z = 0; z < ns; ++z) {
                            psi[t][x][y][z] =
                            +(phi[tup[t]][x][y][z]-2.0*phi[t][x][y][z]+phi[tdn[t]][x][y][z])/gamma
                            -(phi[t][sup[x]][y][z]-2.0*phi[t][x][y][z]+phi[t][sdn[x]][y][z])*gamma
                            -(phi[t][x][sup[y]][z]-2.0*phi[t][x][y][z]+phi[t][x][sdn[y]][z])*gamma
                            -(phi[t][x][y][sup[z]]-2.0*phi[t][x][y][z]+phi[t][x][y][sdn[z]])*gamma
                            +m2*phi[t][x][y][z]-I*epsilon*phi[t][x][y][z]
                            +n*lambda*phi[t][x][y][z]*phi[t][x][y][z]*phi[t][x][y][z];
                            if (cabs(psi[t][x][y][z]) > kmax_thread[omp_get_thread_num()]) kmax_thread[omp_get_thread_num()] = cabs(psi[t][x][y][z]);
                        }
                    }
                }
            }
            double kmax = 0;
            for (int i = 0; i < omp_get_max_threads(); ++i) {
                if (kmax_thread[i] > kmax) kmax = kmax_thread[i];
            }
            #pragma omp parallel for
            for (int t = 0; t < nt; ++t) {
                for (int x = 0; x < ns; ++x) {
                    for (int y = 0; y < ns; ++y) {
                        for (int z = 0; z < ns; ++z) {
                            double u1 = genrand64_real3(&rng[omp_get_thread_num()]);
                            double u2 = genrand64_real3(&rng[omp_get_thread_num()]);
                            double nor = sqrt(-2*log(u1))*cos(2*M_PI*u2);
                            phi[t][x][y][z] += -I*(step/kmax)*psi[t][x][y][z]+sqrt(2*step/kmax)*nor;
                        }
                    }
                }
            }
            if (kmax < bound) {
                #pragma omp parallel for reduction(+:cor[:nt])
                for (int tp = 0; tp < nt; ++tp) {
                    for (int t = 0; t < nt; ++t) {
                        for (int x = 0; x < ns; ++x) {
                            for (int y = 0; y < ns; ++y) {
                                for (int z = 0; z < ns; ++z) {
                                    cor[t] += phi[tp][x][y][z]*phi[(t+tp)%nt][x][y][z]/(ns*ns*ns*nt);
                                }
                            }
                        }
                    }
                }
                ++count;
            }
            time += step/kmax;
            printf("%lf\t%lf\n",time,kmax);
        }
        for (int k = -nt/2; k < nt/2; ++k) {
            double complex sum = 0.0 + 0.0 * I;
            for (int t = 0; t < nt; ++t) {
                sum += cor[t]*cexp(-I*2.0*M_PI*k*t/nt)/count;
            }
            prop[k+nt/2][n] = sum;
            invprop[k+nt/2][n] = 1.0/sum;
        }
    }
    f = fopen("propagator_free.txt", "w");
    for (int k = -nt/2; k < nt/2; ++k) {
        fprintf(f,"%.15f\t%.15f\t%.15f\n",2*M_PI*k/(nt*dt),creal(prop[k+nt/2][0]),cimag(prop[k+nt/2][0]));
    }
    fclose(f);
    f = fopen("propagator_interacting.txt", "w");
    for (int k = -nt/2; k < nt/2; ++k) {
        fprintf(f,"%.15f\t%.15f\t%.15f\n",2*M_PI*k/(nt*dt),creal(prop[k+nt/2][1]),cimag(prop[k+nt/2][1]));
    }
    fclose(f);
    f = fopen("self_energy.txt", "w");
    for (int k = -nt/2; k < nt/2; ++k) {
        fprintf(f,"%.15f\t%.15f\t%.15f\n",2*M_PI*k/(nt*dt),creal(I*(invprop[k+nt/2][0]-invprop[k+nt/2][1])),cimag(I*(invprop[k+nt/2][0]-invprop[k+nt/2][1])));
    }
    fclose(f);
    for (int t = 0; t < nt; ++t) {
        free(invprop[t]);
        free(prop[t]);
    }
    free(invprop);
    free(prop);
    for (int t = 0; t < nt; ++t) {
        for (int x = 0; x < ns; ++x) {
            for (int y = 0; y < ns; ++y) {
                free(phi[t][x][y]);
                free(psi[t][x][y]);
            }
            free(phi[t][x]);
            free(psi[t][x]);
        }
        free(phi[t]);
        free(psi[t]);
    }
    free(phi);
    free(psi);
    free(cor);
    free(seed);
    free(kmax_thread);
    free(rng);
    free(sup);
    free(sdn);
    free(tup);
    free(tdn);
}
