# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The environment class for MonoBeast."""

import torch
import gym


class Environment:
    def __init__(self, gym_env : gym.Env, append_info : bool = False):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None
        self.append_info = append_info

    @property
    def observation_space(self):
        return self.gym_env.observation_space

    @property
    def action_space(self):
        return self.gym_env.action_space

    def seed(self, *args, **kwargs):
        return self.gym_env.seed(*args, **kwargs)

    def format_observations(self, observations) -> dict:
        frame = torch.from_numpy(observations)
        return dict(frame=frame.view((1, 1) + frame.shape))

    def initial(self) -> dict:
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        initial_observations = self.format_observations(self.gym_env.reset())
        return dict(
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
            **initial_observations
        )

    def step(self, action) -> dict:
        observations, reward, done, info = self.gym_env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            observations = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        observations = self.format_observations(observations)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        return_dict = dict(
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
            **observations
        )

        if self.append_info:
            return_dict['info'] = info

        return return_dict

    def close(self):
        self.gym_env.close()


class NetHackEnvironment(Environment):
    """Turns a Gym environment into something that can be step()ed indefinitely."""

    def format_observations(self, observations):
        # print(observations.keys())
        keys=("glyphs", "blstats", "chars")
        new_observations = {}
        for key in keys:
            entry = observations[key]
            entry = torch.from_numpy(entry)
            entry = entry.view((1, 1) + entry.shape)  # (...) -> (T,B,...).
            new_observations[key] = entry
        return new_observations
