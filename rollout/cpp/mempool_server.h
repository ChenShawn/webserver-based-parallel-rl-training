#ifndef _MEMPOOL_SERVER_H_
#define _MEMPOOL_SERVER_H_

#include <gflags/gflags.h>
#include <butil/logging.h>
#include <brpc/server.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <atomic>
#include <thread>

#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "samples.pb.h"

// Date type for each element in the memory buffer, either float or double
// Must be compatible with that defined in samples.proto
#define DATA_TYPE float

DECLARE_int32(port);
DECLARE_int32(idle_timeout_s);
DECLARE_int32(batch_size);
DECLARE_int32(state_size);
DECLARE_int32(action_size);
DECLARE_int32(shmkey);
DECLARE_int32(replay_capacity);
DECLARE_int32(safety_size);


template <typename DType>
class ReplayMemory {
public:
    ReplayMemory();
    ReplayMemory(const ReplayMemory&) = delete;
    ~ReplayMemory();

    int push(const Episode* episode);
    // Don't use networking for sampling. It's so fucking stupid!
    int sample(int batch_size);
    int close();

    inline int get_size() const { return size.load(); }
    inline int get_num_read() const { return num_read.load(); }
    inline int get_num_write() const { return num_write.load(); }
    inline uint64_t get_num_samples() const { return shared_ptr.load(); }
    inline int get_shmid() const { return shmid; }

    void copy_from_pbdata(DType* ptr, const Sample& sp);

private:
    DType* buffer;
    std::atomic_uint64_t shared_ptr;

    // variables about shared memory
    DType* shmbuf;
    int shmid;
    // statistics
    std::atomic_uint32_t size;
    std::atomic_uint32_t num_read, num_write;
};

#endif