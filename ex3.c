#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <arm_neon.h>
#include <time.h>

#define NUM_THREADS 4


//gloabal declaratons 
int num = 100000000;

float *a;
float *b;
float *r;


void* mult_vect(void* threadId)
{
        float32x4_t va, vb, vr;

        for( int i = ((num/NUM_THREADS) * (long)threadId); i < (num/NUM_THREADS) * ((long)threadId + 1); i++ )
	{
		va = vld1q_f32(&a[i]);
		vb = vld1q_f32(&b[i]);

		vr = vmulq_f32(va, vb);

		vst1q_f32(&r[i], vr);                
	}
        pthread_exit(NULL);
        return NULL; // this line is not reached due to pthread_exit
	
}

void* mult_std(void* threadId)
{
	for( int i = ((num/NUM_THREADS) * (long)threadId); i < (num/NUM_THREADS) * ((long)threadId + 1); i++ )
	{
                r[i] = a[i] * b[i];
	}
        pthread_exit(NULL);
        return NULL; // this line is not reached due to pthread_exit
}

int main (int argc, char *argv[])
{
        pthread_t threads[NUM_THREADS];
        int rc;

        a = (float*)aligned_alloc(16, num*sizeof(float));
        b = (float*)aligned_alloc(16, num*sizeof(float));
        r = (float*)aligned_alloc(16, num*sizeof(float));



        for(int i = 0; i < num; i++)
	{
		a[i] = (i % 127)*0.1457f;
		b[i] = (i % 331)*0.1231f;
	}

        struct timespec ts_start;
	struct timespec ts_end;

        clock_gettime(CLOCK_MONOTONIC, &ts_start);

        
        printf("number of threads: %d\n", NUM_THREADS);
        for(long j = 0; j < NUM_THREADS; j++)
        {
                //printf("Working on thread %ld\n", j);
                //rc = pthread_create(&threads[j], NULL, mult_std, (void*)j);
                rc = pthread_create(&threads[j], NULL, mult_std, (void*)j);
                if(rc)
                {
                        printf("Error: Unable to create and work on thread, %d\n", rc);
                        exit(-1);
                }
        }

        clock_gettime(CLOCK_MONOTONIC, &ts_end);

        double duration_std = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) * 1e-9;


        printf("Elapsed time std: %f\n", duration_std);

         //Join the threads
	for(long t = 0; t < NUM_THREADS; t++)
	{
		pthread_join(threads[t], NULL);
	}

        int errors = 0;
        for (int i = 0; i < num; i++)
        {
                if ( (r[i] - a[i]*b[i]) > 0.0001)
                        errors++;
        }

        printf("Errors: %d\n", errors);
        free(a);
        free(b);
        free(r);

        return 0;
}


