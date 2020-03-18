#include <cstdio>
#include <cstdlib>
#include <memory>
#include <gflags/gflags.h>
#include <butil/logging.h>
#include <brpc/server.h>
#include <brpc/channel.h>

#include "Python.h"

// pretend that we are using gflags
const char FLAGS_protocol[] = "baidu_std";
const char FLAGS_remote_addr[] = "0.0.0.0:8000";
const char FLAGS_connection_type[] = "single";
const char FLAGS_load_balancer[] = "";
brpc::Channel* channel;
MempoolService_Stub* stub;

extern "C" {
// Python-C++ interface must be compiled as extern "C"

// Only called once in the beginning of the program
void init_channel(int timeout_ms, int max_retry) {
    // TODO: not sure if new operator works here...
    channel = new brpc::Channel();
    // channel = (brpc::Channel*) malloc(sizeof(brpc::Channel));
    // std::allocator<brpc::Channel> alloc_chnl;
    // alloc_chnl.construct(channel);
    brpc::ChannelOptions options;
    options.protocol = FLAGS_protocol;
    options.connection_type = FLAGS_connection_type;
    options.timeout_ms = timeout_ms;
    options.max_retry = max_retry;
    if (channel->Init(FLAGS_remote_addr, FLAGS_load_balancer, &options) != 0) {
        LOG(ERROR) << "Failed to initialize channel";
        exit(-1);
    }
    // stub = (MempoolService_Stub*) malloc(sizeof(MempoolService_Stub));
    // std::allocator<MempoolService_Stub> alloc_stub;
    // alloc_stub.construct(stub, channel);
    stub = new MempoolService_Stub(channel);
}

void send_rpc_request(int n_samples, int state_size, int action_size,
                      float* states, float* actions, float* rewards, 
                      float* next_states, float* masks) {
    // release python GIL to allow sending samples asynchronously
    Py_BEGIN_ALLOW_THREADS;

    for (int i = 0; i < n_samples ; i ++) {
        ;
    }

    Py_END_ALLOW_THREADS;
}

// Only called once in the ending of the program
void close_channel() {
    delete channel;
    delete stub;
}


} // extern "C"
