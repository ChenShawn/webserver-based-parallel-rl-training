#include "mempool_server.h"

/*******
 * Compile linking:
 * g++ -L/usr/lib/x86_64-linux-gnu -L/home/yuki/Downloads/incubator-brpc/bld/output/lib \
 * -Xlinker "-(" samples.pb.o mempool_server.o -Wl,-Bstatic -lgflags -lprotobuf -lbrpc \
 * -Wl,-Bdynamic -Xlinker "-)" -lpthread -lssl -lcrypto -ldl -lz -lrt -lleveldb -o mempool_server
 * 
*******/

DEFINE_int32(port, 20000, "TCP Port of this server");
DEFINE_int32(idle_timeout_s, -1, "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s`. "
             "By default set as -1 don't close any connection");
DEFINE_string(actor_name, "/home/yuki/Documents/mycode/micro-wookong/train/train/Pendulum-v0/sac_actor.pth",
              "Directory of the actor model file");
DEFINE_int32(batch_size, 512, "batch size for mempool reading operations");
DEFINE_int32(state_size, 12, "total dimension of state features");
DEFINE_int32(action_size, 4, "total dimension of actions");
DEFINE_int32(shmkey, 654321, "non-nagative intergers to uniquely identify the shmid");
DEFINE_int32(replay_capacity, 1048576, "maximum capacity of replay memory");
DEFINE_int32(safety_size, 5000, "identifying safety zone size in replay memory to avoid data race");


template <typename DType>
ReplayMemory<DType>::ReplayMemory() {
    // initialize sample buffer
    int sample_size = sizeof(DType) * (FLAGS_state_size * 2 + FLAGS_action_size + 2);
    this->buffer = (DType*)malloc(sample_size * FLAGS_replay_capacity);
    // initialize shared memory buffer
    int shmsize = sample_size * FLAGS_batch_size;
    this->shmid = shmget((key_t)FLAGS_shmkey, shmsize, 0666|IPC_CREAT);
    if (this->shmid == -1) {
        LOG(ERROR) << "create shm failed, maybe run `ipcs -m` to check...";
        exit(-1);
    }
    this->shmbuf = (DType*)shmat(this->shmid, NULL, 0);
    this->shared_ptr = 0;
    if ((long long)shmbuf == -1) {
        LOG(ERROR) << "shmat failed";
        exit(-1);
    }
    this->size = 0;
    this->num_read = 0;
    this->num_write = 0;
}

template <typename DType>
ReplayMemory<DType>::~ReplayMemory() {
    this->close();
    if (this->buffer != NULL) {
        free(this->buffer);
        this->buffer = NULL;
    }
}

template <typename DType>
int ReplayMemory<DType>::close() {
    if (this->shmid != -1) {
        if (shmdt(this->shmbuf) == -1) {
            LOG(ERROR) << "shmdt failed";
            return -1;
        }
        if (shmctl(this->shmid, IPC_RMID, 0) == -1) {
            LOG(ERROR) << "remove shm error";
            return -2;
        }
        LOG(INFO) << "shared memory " << shmid << " has been released";
        this->shmid = -1;
    }
    return 0;
}

template <typename DType>
int ReplayMemory<DType>::push(const Episode* episode) {
    // use atomic operation to avoid data race
    int sample_size = 2 * FLAGS_state_size * 2 + FLAGS_action_size + 2;
    for (int i = 0; i < episode->samples_size(); i ++) {
        uint64_t write_index = this->shared_ptr.fetch_add(1);
        if (write_index >= (uint64_t)FLAGS_replay_capacity) {
            write_index %= FLAGS_replay_capacity;
        }
        copy_from_pbdata(this->buffer + i * sample_size, episode->samples(i));
        this->size += 1;
    }
    this->num_write += 1;
    LOG(DEBUG) << "push an episode of samples with size=" << episode->samples_size();
    return 0;
}

template <typename DType>
int ReplayMemory<DType>::sample(int batch_size) {
    if (batch_size > FLAGS_batch_size) {
        batch_size = FLAGS_batch_size;
    }
    int cursize = this->get_size();
    if (batch_size >= cursize) {
        LOG(ERROR) << "Trying to read a samples of batch_size="
                   << batch_size << " but replay memory has size="
                   << cursize;
        return -1;
    }
    int sample_size = FLAGS_state_size * 2 + FLAGS_action_size + 2;
    int read_count = 0, rand_idx, curpos, endpos;
    srand((unsigned)time(NULL));
    while (read_count < batch_size) {
        curpos = this->shared_ptr.load() % FLAGS_replay_capacity;
        endpos = (curpos + FLAGS_safety_size) % FLAGS_replay_capacity;
        rand_idx = rand() % this->size.load();
        if (rand_idx >= curpos && rand_idx < endpos)
            continue;
        // copy samples to shared memory buffer
        memcpy(this->shmbuf + read_count * sample_size, 
               this->buffer + rand_idx * sample_size, 
               sample_size);
        read_count ++;
    }
    this->num_read += 1;
    LOG(DEBUG) << "load samples with batch_size=" << batch_size;
    return 0;
}

template <typename DType>
void ReplayMemory<DType>::copy_from_pbdata(DType* ptr, const Sample& sp) {
    for (int i = 0; i < FLAGS_state_size; i ++) {
        ptr[i] = sp.state(i);
    }
    for (int i = 0; i < FLAGS_action_size; i ++) {
        ptr[FLAGS_state_size + i] = sp.action(i);
    }
    ptr[FLAGS_state_size + FLAGS_action_size] = sp.reward();
    for (int i = 0; i < FLAGS_state_size; i ++) {
        ptr[FLAGS_state_size + FLAGS_action_size + 1 + i] = sp.next_state(i);
    }
    ptr[FLAGS_state_size * 2 + FLAGS_action_size + 1] = sp.mask();
}


template <typename T>
class MempoolServiceImpl : public MempoolService {
public:
    MempoolServiceImpl(): mem_ptr(NULL) {}
    MempoolServiceImpl(ReplayMemory<T>* _mem_ptr): mem_ptr(_mem_ptr) {}
    virtual ~MempoolServiceImpl() {}

    virtual void HelloWorld(google::protobuf::RpcController* cntl_base,
                            const EmptyRequest* request,
                            StatusResponse* response,
                            google::protobuf::Closure* done) {
        brpc::ClosureGuard done_guard(done);
        brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
        LOG(DEBUG) << "Received hello world request from " << cntl->remote_side();
        // Fill response
        response->set_status(200);
        response->set_error_text("Hello world!");
        // cntl->set_response_compress_type(brpc::COMPRESS_TYPE_GZIP);
    }

    virtual void SaveSamples(google::protobuf::RpcController* cntl_base,
                             const ::DRL::Episode* episode,
                             ::DRL::StatusResponse* response, 
                             google::protobuf::Closure* done) {
        brpc::ClosureGuard done_guard(done);
        brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
        LOG(DEBUG) << "Received save samples request from " << cntl->remote_side();
        // save samples
        mem_ptr->push(episode);
        response->set_status(200);
    }

    virtual void ReadSamples(google::protobuf::RpcController* cntl_base,
                             const ::DRL::ReadRequest* request,
                             ::DRL::StatusResponse* response,
                             google::protobuf::Closure* done) {
        brpc::ClosureGuard done_guard(done);
        brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
        LOG(DEBUG) << "Received reading request from " << cntl->remote_side();
        // load samples to shmbuf
        if (mem_ptr->sample(request->batch_size()) != 0) {
            response->set_status(500);
            response->set_error_text("mempool not sufficiently filled...");
        } else {
            response->set_status(200);
        }
    }

    virtual void CloseShmBuffer(google::protobuf::RpcController* cntl_base,
                                const ::DRL::EmptyRequest* request,
                                ::DRL::StatusResponse* response,
                                ::google::protobuf::Closure* done) {
        brpc::ClosureGuard done_guard(done);
        brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
        if (mem_ptr->close() != 0) {
            response->set_status(500);
            response->set_error_text("either shmdt or shmctl is failed, check logs for more info");
        } else {
            response->set_status(200);
        }
        LOG(DEBUG) << "Shared memory shut down request from " << cntl->remote_side();
    }

    virtual void GetStatInfo(google::protobuf::RpcController* cntl_base,
                             const ::DRL::EmptyRequest* request,
                             ::DRL::StatInfoResponse* response,
                             ::google::protobuf::Closure* done) {
        brpc::ClosureGuard done_guard(done);
        brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
        response->set_num_reads(mem_ptr->get_num_read());
        response->set_num_writes(mem_ptr->get_num_write());
        response->set_num_samples(mem_ptr->get_num_samples());
        LOG(DEBUG) << "Receive GetStatInfo request from " << cntl->remote_side();
    }

    virtual void GetShmInfo(google::protobuf::RpcController* cntl_base,
                            const ::DRL::EmptyRequest* request,
                            ::DRL::ShmInfoResponse* response,
                            ::google::protobuf::Closure* done) {
        brpc::ClosureGuard done_guard(done);
        brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
        response->set_shmid(this->mem_ptr->get_shmid());
        response->set_offset(0);
        LOG(DEBUG) << "Receive GetShmInfo request from " << cntl->remote_side();
    }

    virtual void DownloadModelFiles(google::protobuf::RpcController* cntl_base,
                        const ::DRL::EmptyRequest* request,
                        ::DRL::StatusResponse* response,
                        ::google::protobuf::Closure* done) {
        brpc::ClosureGuard done_guard(done);
        brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
        FILE *fin = NULL;
        size_t fsize = 0;
        fin = fopen(FLAGS_actor_name, "rb");
        if (fin == NULL) {
            LOG(ERROR) << "Local model file [" << FLAGS_actor_name << "] does not existed";
            response->set_status(500);
            response->set_error_text("Local model file does not existed!");
        } else {
            fseek(fin, 0, SEEK_END);
            fsize = ftell(fin);
            rewind(fin);
            char *tmpbuf = (char*)malloc(fsize);
            if (fsize <= 0 || tmpbuf == NULL) {
                LOG(ERROR) << "BadAlloc when trying to allocate " << fsize << " bytes";
                fclose(fin);
                return;
            } else {
                fread(tmpbuf, 1, fsize, fin);
                fclose(fin);
            }
            cntl->response_attachment().append(tmpbuf, fsize);
            response->set_status(200);
        }
    }

private:
    ReplayMemory<T>* mem_ptr;
};


int main(int argc, char* argv[]) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    GFLAGS_NS::ParseCommandLineFlags(&argc, &argv, true);
    brpc::Server server;
    ReplayMemory<DATA_TYPE> replay_memory;
    MempoolServiceImpl<DATA_TYPE> mempool_service_impl(&replay_memory);

    // Add the service into server. Notice the second parameter, because the
    // service is put on stack, we don't want server to delete it, otherwise
    // use brpc::SERVER_OWNS_SERVICE.
    if (server.AddService(&mempool_service_impl, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
        LOG(ERROR) << "Fail to add service";
        return -1;
    }

    // Start the server.
    brpc::ServerOptions options;
    options.idle_timeout_sec = FLAGS_idle_timeout_s;
    if (server.Start(FLAGS_port, &options) != 0) {
        LOG(ERROR) << "Fail to start EchoServer";
        return -1;
    }

    // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
    server.RunUntilAskedToQuit();
    return 0;
}
