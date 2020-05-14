#include <cstdio>
#include <cstdlib>
#include <memory>
#include <gflags/gflags.h>
#include <butil/logging.h>
#include <brpc/server.h>
#include <brpc/channel.h>
#include <brpc/stream.h>
#include <unistd.h>
#include <Python.h>

#include "samples.pb.h"

brpc::Channel* channel;
MempoolService_Stub* stub;

extern "C" {
// Python-C++ interface must be compiled as extern "C"

void init_channel(char* remote_addr, int timeout_ms, int max_retry);
void send_rpc_request(int n_samples, int state_size, int action_size,
                      float* states, float* actions, float* rewards, 
                      float* next_states, float* masks);
void download_model_files();
void close_channel();

} // extern "C"


/***********************************
 * 
 * FUNCTION IMPLMENTATION GOES BELOW...
 * 
 * *******************************/

// Only called once in the beginning of the program
void init_channel(char* remote_addr, int timeout_ms, int max_retry) {
    // TODO: not sure if new operator works here...
    channel = new brpc::Channel();
    brpc::ChannelOptions options;
    options.protocol = "baidu_std";
    options.connection_type = "single";
    options.timeout_ms = timeout_ms;
    options.max_retry = max_retry;
    if (channel->Init(FLAGS_remote_addr, "", &options) != 0) {
        LOG(ERROR) << "Failed to initialize channel";
        delete channel;
        exit(-1);
    }
    stub = new DRL::MempoolService_Stub(channel);
}


// Send an RPC request to save an episode of samples
// formed as [s, a, r, s_next, masks] of type float32
void send_rpc_request(int n_samples, int state_size, int action_size,
                      float* states, float* actions, float* rewards, 
                      float* next_states, float* masks) {
    // release python GIL to allow sending samples asynchronously
    Py_BEGIN_ALLOW_THREADS;

    // Put request on heap 
    DRL::Episode request;
    DRL::StatusResponse response;
    int sample_size = state_size * 2 + action_size + 2;
    for (int i = 0; i < n_samples ; i ++) {
        DRL::Sample* sample = request.add_samples();
        for (int j = 0; j < state_size; j ++) {
            sample->add_state(states[i * state_size + j]);
            sample->add_next_state(next_states[i * state_size + j]);
        }
        for (int j = 0; j < action_size; j ++) {
            sample->add_action(actions[i * action_size + j]);
        }
        sample->set_mask(masks[i]);
        sample->set_reward(rewards[i]);
    }
    // synchronous request: set the last param to NULL
    brpc::Controller cntl;
    stub->SaveSamples(&cntl, &request, &response, NULL);
    if (cntl.Failed() || response.status != 200) {
        LOG(ERROR) << "Save model failed -- ErrorText: " 
                   << cntl.ErrorText()
                   << " -- StatusCode: "
                   << response.status;
    }

    Py_END_ALLOW_THREADS;
}


// Send a request to mempool server and download model files via attachment
void download_model_files(char* actor_name) {
    // release python GIL to allow sending samples asynchronously
    Py_BEGIN_ALLOW_THREADS;

    brpc::Controller cntl;
    DRL::EmptyRequest request;
    DRL::StatusResponse response;

    stub->DownloadModelFiles(&cntl, &request, &response, NULL);
    // remote return status 200 for new model files
    if ((!cntl.Failed()) && response.status == 200) {
        FILE *fout = NULL;
        fout = fopen(actor_name, "wb+");
        if (fout == NULL) {
            LOG(ERROR) << "Failed to open file " << std::string(actor_name);
            return;
        }
        const void *fdata = cntl.response_attachment().fetch1();
        if (fdata != NULL) {
            fwrite(fdata, 1, cntl.response_attachment().size(), fdata);
        } else {
            LOG(ERROR) << "Remote response has no attachment!";
        }
        fclose(fout);
    } else {
        LOG(ERROR) << "Download model file from " << cntl.remote_side() << " to "
                   << cntl.local_side() << " has failed -- " << cntl.ErrorText()
                   << " -- StatusCode=" << response.status;
    }
    
    Py_END_ALLOW_THREADS;
}


// Only called once in the end of the program
void close_channel() {
    delete channel;
    delete stub;
    channel = NULL;
    stub = NULL;
}