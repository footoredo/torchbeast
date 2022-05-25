/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <tuple>

#include <mutex>
#include <random>

#include <grpc++/grpc++.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nest_serialize.h"
#include "rpcenv.grpc.pb.h"
#include "rpcenv.pb.h"

#include "nest/nest.h"
#include "nest/nest_pybind.h"

namespace py = pybind11;

typedef nest::Nest<py::array> PyArrayNest;

namespace rpcenv {
class EnvServer {
 private:
  class ServiceImpl final : public RPCEnvServer::Service {
   public:
    ServiceImpl(py::object env_init, int server_id, bool done_at_reward, int num_labels) : 
      env_init_(env_init), 
      server_id_(server_id),
      done_at_reward_(done_at_reward),
      num_labels_(num_labels),
      mutexes_(),
      rng_(),
      rng_mutex_() {
        std::vector<std::mutex> mutexes(num_labels_);
        mutexes_.swap(mutexes);
        // if (!env_datas_initialized_) {
        //   env_datas_ = std::make_shared<std::vector<std::deque<PyArrayNest>>>(num_labels_);
        //   env_datas_initialized_ = true;
        //   // for (int i = 0; i < num_labels_; ++ i) {
        //   //   env_datas_->push_back(std::deque<PyArrayNest>());
        //   // }
        // }
      }

   private:
    virtual grpc::Status StreamingEnv(
        grpc::ServerContext *context,
        grpc::ServerReaderWriter<Step, Action> *stream) override {
      py::gil_scoped_acquire acquire;  // Destroy after pyenv.
      py::object pyenv;
      py::object stepfunc;
      py::object resetfunc;
      py::object completefunc;
      py::object dataresetfunc;
      py::object serializefunc;

      // static std::mutex mutex;
      // std::unique_lock<std::mutex> lock(mutex);

      // static std::vector<std::deque<PyArrayNest>> env_datas_(num_labels_);
      // static int counter = 0;

      // counter += 1;

      // std::cout << "Server id: " << server_id_ << " counter: " << counter << std::endl;

      // lock.unlock();

      PyArrayNest observation;
      PyArrayNest last_observation;
      PyArrayNest init_observation;
      float reward = 0.0;
      bool done = true;
      bool delayed_done = false;
      bool reset_done = true;
      int episode_step = 0;
      float episode_return = 0.0;
      int episode_count = 0;
      bool done_at_reward = done_at_reward_;
      py::array env_data;
      std::unique_lock<std::mutex> rng_lock(rng_mutex_);
      rng_lock.unlock();
      // std::hash<std::string> hasher;
      // int episode_hash = hasher(server_address_ + "#" + std::to_string(episode_count_));

      auto set_observation = py::cpp_function(
          [&observation](PyArrayNest o) { observation = std::move(o); },
          py::arg("observation"));

      auto set_observation_reward_done = py::cpp_function(
          [&observation, &reward, &done, &reset_done, done_at_reward](PyArrayNest o, float r, bool d,
                                         py::args) {
            observation = std::move(o);
            reward = r;
            done = d || (done_at_reward && r > 0.1);
            reset_done = d;
          },
          py::arg("observation"), py::arg("reward"), py::arg("done"));

      auto set_env_data = py::cpp_function(
          [&env_data](py::array d) { env_data = std::move(d); },
          py::arg("env_data"));

      // auto data_reset = py::cpp_function([](bool ret_val) {});
      // auto complete_objective = py::cpp_function([]() {});

      try {
        pyenv = env_init_();
        stepfunc = pyenv.attr("step");
        resetfunc = pyenv.attr("reset");
        completefunc = pyenv.attr("complete_objective");
        dataresetfunc = pyenv.attr("set_next_reset_config");
        serializefunc = pyenv.attr("serialize");
        set_observation(resetfunc());
      } catch (const pybind11::error_already_set &e) {
        // Needs to be caught and not re-raised, as this isn't in a Python
        // thread.
        std::cerr << e.what() << std::endl;
        return grpc::Status(grpc::INTERNAL, e.what());
      }

      Step step_pb;
      init_observation = PyArrayNest(observation.value);
      last_observation = PyArrayNest(observation.value);
      fill_nest_pb(step_pb.mutable_observation(), std::move(observation), fill_ndarray_pb);
      fill_nest_pb(step_pb.mutable_last_observation(), std::move(init_observation), fill_ndarray_pb);  // Placeholder

      step_pb.set_reward(reward);
      step_pb.set_done(done);
      step_pb.set_reset_done(reset_done);
      step_pb.set_episode_step(episode_step);
      step_pb.set_episode_return(episode_return);
      step_pb.set_episode_server(server_id_);
      step_pb.set_episode_count(episode_count);

      Action action_pb;
      while (true) {
        {
          py::gil_scoped_release release;  // Release while doing transfer.
          stream->Write(step_pb);
          if (!stream->Read(&action_pb)) {
            break;
          }
        }
        try {
          step_pb.Clear();

          fill_nest_pb(step_pb.mutable_last_observation(), std::move(last_observation), fill_ndarray_pb);

          // I'm not sure if this is fast, but it's convienient.
          set_observation_reward_done(*stepfunc(nest_pb_to_nest(
              action_pb.mutable_nest_action(), array_pb_to_nest)));
          last_observation = PyArrayNest(observation.value);

          episode_step += 1;
          episode_return += reward;

          // if (~action_pb.save_env()) {
          if (true) {
            int task_id = action_pb.save_env() / 16;
            // if (delayed_done) {
            //   done = true;
            // }
            if (action_pb.save_env() & 1) {
              done = true;
              reset_done = true;
            }
            if (action_pb.save_env() & 2) {
              set_env_data(serializefunc());
              // int label = action_pb.save_env();
              // std::lock_guard<std::mutex> lock(mutexes_[label]);
              // env_datas_[label].push_back(env_data);
              
              // std::cout << "server saving env!!" << std::endl;

              fill_ndarray_pb(step_pb.mutable_env_data(), std::move(env_data));
            }
            if (action_pb.save_env() & 4) {
              completefunc(task_id, (action_pb.save_env() & 8) > 0);
            }
            if (action_pb.save_env() & 8) {
              // delayed_done = true;
              done = true;
            }
            // else {
            //   delayed_done = false;
            // }
          }

          step_pb.set_reward(reward);
          step_pb.set_done(done);
          step_pb.set_reset_done(reset_done);
          step_pb.set_episode_step(episode_step);
          step_pb.set_episode_return(episode_return);
          step_pb.set_episode_server(server_id_);
          step_pb.set_episode_count(episode_count);

          // if (episode_step % 50 == 0) {
          //   set_env_data(serializefunc());
          //   fill_ndarray_pb(step_pb.mutable_env_data(), std::move(env_data));
          // }

          if (reset_done) {
            // Reset episode_* for the _next_ step.
            episode_step = 0;
            episode_return = 0.0;
            episode_count += 1;
            // episode_hash = hasher(server_address_ + "#" + std::to_string(episode_count_));
          }
          if (reset_done) {
            // rng_lock.lock();
            // int label = std::uniform_int_distribution<int>(0, num_labels_)(rng_) - 1;
            // // std::cout << "loading env!! label:" << label << std::endl;
            // rng_lock.unlock();
            // if (~label) {
            //   std::lock_guard<std::mutex> lock(mutexes_[label]);
            //   // std::cout << "loading env!! label:" << label << " length " << env_datas_[label].size() << std::endl;
            //   if (env_datas_[label].size() > 0) {
            //     rng_lock.lock();
            //     int index = std::uniform_int_distribution<int>(0, env_datas_[label].size() - 1)(rng_);
            //     rng_lock.unlock();
            //     dataresetfunc(label, std::move(env_datas_[label][index]));
            //   }
            // }

            if (action_pb.has_env_data()) {
              // std::cout << "server loading env!! label: " << action_pb.env_data_label() << std::endl;
              dataresetfunc(action_pb.env_data_label(), array_pb_to_nest(action_pb.mutable_env_data()));
            }
            
            set_observation(resetfunc());
            last_observation = PyArrayNest(observation.value);
          }
        } catch (const pybind11::error_already_set &e) {
          std::cerr << e.what() << std::endl;
          return grpc::Status(grpc::INTERNAL, e.what());
        }

        fill_nest_pb(step_pb.mutable_observation(), std::move(observation), fill_ndarray_pb);
      }
      return grpc::Status::OK;
    }

    py::object env_init_;  // TODO: Make sure GIL is held when destroyed.
    int server_id_;
    bool done_at_reward_;
    int num_labels_;
    std::vector<std::mutex> mutexes_;
    // static std::shared_ptr<std::vector<std::deque<PyArrayNest>>> env_datas_;
    std::mt19937 rng_;
    mutable std::mutex rng_mutex_;

    // TODO: Add observation and action size functions (pre-load env)
  };

 public:
  EnvServer(py::object env_class, const std::string &server_address, int server_id, bool done_at_reward, int num_labels)
      : server_address_(server_address),
        server_id_(server_id),
        service_(env_class, server_id, done_at_reward, num_labels),
        server_(nullptr) {}

  void run() {
    if (server_) {
      throw std::runtime_error("Server already running");
    }
    py::gil_scoped_release release;

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address_,
                             grpc::InsecureServerCredentials());
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    std::cerr << "Server listening on " << server_address_ << std::endl;

    server_->Wait();
  }

  void stop() {
    if (!server_) {
      throw std::runtime_error("Server not running");
    }
    server_->Shutdown();
  }

  static void fill_ndarray_pb(rpcenv::NDArray *array, py::array pyarray) {
    // Make sure array is C-style contiguous. If it isn't, this creates
    // another memcopy that is not strictly necessary.
    if ((pyarray.flags() & py::array::c_style) == 0) {
      pyarray = py::array::ensure(pyarray, py::array::c_style);
    }

    // This seems surprisingly involved. An alternative would be to include
    // numpy/arrayobject.h and use PyArray_TYPE.
    int type_num =
        py::detail::array_descriptor_proxy(pyarray.dtype().ptr())->type_num;

    // if (type_num != 2) {
    //   std::cout << "server side: " << pyarray.dtype() << " " << type_num << std::endl;
    // }

    array->set_dtype(type_num);
    for (size_t i = 0, ndim = pyarray.ndim(); i < ndim; ++i) {
      array->add_shape(pyarray.shape(i));
    }

    // TODO: Consider set_allocated_data.
    // TODO: consider [ctype = STRING_VIEW] in proto file.
    py::buffer_info info = pyarray.request();
    array->set_data(info.ptr, info.itemsize * info.size);
  }

  static PyArrayNest array_pb_to_nest(rpcenv::NDArray *array_pb) {
    std::vector<int64_t> shape;
    for (int i = 0, length = array_pb->shape_size(); i < length; ++i) {
      shape.push_back(array_pb->shape(i));
    }

    // Somewhat complex way of turning an type_num into a py::dtype.
    py::dtype dtype = py::reinterpret_borrow<py::dtype>(
        py::detail::npy_api::get().PyArray_DescrFromType_(array_pb->dtype()));

    std::string *data = array_pb->release_data();

    // Attach capsule as base in order to free data.
    return PyArrayNest(py::array(dtype, shape, {}, data->data(),
                                 py::capsule(data, [](void *ptr) {
                                   delete reinterpret_cast<std::string *>(ptr);
                                 })));
  }

 private:
  const std::string server_address_;
  const int server_id_;
  ServiceImpl service_;
  std::unique_ptr<grpc::Server> server_;
};

}  // namespace rpcenv

void init_rpcenv(py::module &m) {
  py::class_<rpcenv::EnvServer>(m, "Server")
      .def(py::init<py::object, const std::string &, int, bool, int>(), py::arg("env_class"),
           py::arg("server_address") = "unix:/tmp/polybeast", py::arg("server_id") = 0, 
           py::arg("done_at_reward") = false, py::arg("num_labels") = 1)
      .def("run", &rpcenv::EnvServer::run)
      .def("stop", &rpcenv::EnvServer::stop);
}
