#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <errno.h>


extern "C" {

void init_shm(float* shmptr, int _shmsz, int _shmkey, int* additional) {
    int shmid = -1;
    shmid = shmget((key_t)_shmkey, _shmsz, 0666|IPC_CREAT);
    if (shmid == -1) {
        perror("create shm failed, maybe run `ipcs -m` to check...`\n");
        exit(-1);
    }
    void *shm_start_addr = (void*)((char*)shmptr + getpagesize());
    // NOTE: the addr of shmptr may have already been mapped in pgtable
    // the flag SHM_REMAP enforces remapping, which is non-optional
    void *shared_ptr = (void*)shmat(shmid, shm_start_addr, SHM_RND|SHM_REMAP);
    if ((long long)shared_ptr == -1) {
        int err = errno;
        printf("shmat failed, errno=%d\n", err);
        perror("shmat failed\n");
        exit(-1);
    }
    // printf("shmaddr: input %p output %p\n", shmptr, shared_ptr);
    additional[0] = shmid;
    additional[1] = (char*)shared_ptr - (char*)shmptr;
}


void fill_batch_shm(float* shmbuf, int* addinfo, float* state, float* action, 
                    float* reward, float* next_state, float* mask) {
    int shmid = addinfo[0], offset = addinfo[1];
    int state_size = addinfo[2], action_size = addinfo[3], batch_size = addinfo[4];
    int sample_size = state_size * 2 + action_size + 2;
    float* shm_start = (float*)((char*)shmbuf + offset);
    for (int i = 0; i < batch_size; i ++) {
        for (int j = 0; j < state_size; j ++) {
            shm_start[i * sample_size + j] = state[i * state_size + j];
        }
        for (int j = 0; j < action_size; j ++) {
            shm_start[i * sample_size + state_size + j] = action[i * action_size + j];
        }
        shm_start[i * sample_size + state_size + action_size] = reward[i];
        for (int j = 0; j < state_size; j ++) {
            shm_start[i * sample_size + state_size + action_size + 1 + j] = next_state[j];
        }
        shm_start[i * sample_size + 2 * state_size + action_size + 1] = mask[i];
    }
}


/****************************
 * read_batch_shm: this can be seen as the reversed process of fill_batch_shm
 * **************************/
void read_batch_shm(float* shmbuf, int* addinfo, float* state, float* action, 
                    float* reward, float* next_state, float* mask) {
    int shmid = addinfo[0], offset = addinfo[1];
    int state_size = addinfo[2], action_size = addinfo[3], batch_size = addinfo[4];
    int sample_size = state_size + action_size + 1;
    float* shm_start = (float*)((char*)shmbuf + offset);
    for (int i = 0; i < batch_size; i ++) {
        for (int j = 0; j < state_size; j ++) {
            state[i * state_size + j] = shm_start[i * sample_size + j];
        }
        for (int j = 0; j < action_size; j ++) {
            action[i * action_size + j] = shm_start[i * sample_size + state_size + j];
        }
        reward[i] = shm_start[i * sample_size + state_size + action_size];
        for (int j = 0; j < state_size; j ++) {
            next_state[j] = shm_start[i * sample_size + state_size + action_size + 1 + j];
        }
        mask[i] = shm_start[i * sample_size + 2 * state_size + action_size + 1];
    }
}


void close_shm(float* shmbuf, int* addinfo) {
    int shmid = addinfo[0];
    int offset = addinfo[1];
    if (shmdt((char*)shmbuf + offset) == -1) {
        perror("shmdt failed\n");
        exit(-1);
    }
    if (shmctl(shmid, IPC_RMID, 0) == -1) {
        perror("remove shm error\n");
        exit(-1);
    }
}


#ifdef MODE_DEBUG
/***
 * **************
 * ONLY FOR DEBUG
 * **************
 */
int main() {
    float* membuf = (float*)malloc(getpagesize() * 64);
    int shmsz = getpagesize() * 62;
    int addtion[2] = {0, 0};
    int shmkey = 654321;

    init_shm(membuf, shmsz, shmkey, addtion);
    int shmid = addtion[0], offset = addtion[1];
    printf("shmid=%d\n", shmid);
    printf("page aligned shared memory results: input %p offset %d\n", membuf, offset);

    if (shmdt((char*)membuf + offset) == -1) {
        perror("shmdt failed\n");
        exit(-1);
    }
    if (shmctl(shmid, IPC_RMID, 0) == -1) {
        perror("remove shm error\n");
        exit(-1);
    }

    return 0;
}

#endif
    
} // extern "C"
