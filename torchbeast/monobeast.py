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

import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback
import typing
from collections import defaultdict, deque
import joblib

# os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

import gym
import nle

from torchbeast import atari_wrappers
from torchbeast.core import environment
from torchbeast.core import net
from torchbeast.core import file_writer
from torchbeast.core import prof
from torchbeast.core import vtrace


# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--env", type=str, default="PongNoFrameskip-v4",
                    help="Gym environment.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")

parser.add_argument("--loaddir")
parser.add_argument("--no_actor_learning", action="store_true")
parser.add_argument("--no_predictor_learning", action="store_true")
parser.add_argument("--test_episodes", default=10, type=int)

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="~/logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=4, type=int, metavar="N",
                    help="Number of actors (default: 4).")
parser.add_argument("--total_steps", default=100000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--num_learner_threads", "--num_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")

parser.add_argument("--save_ttyrec_every", default=1000, type=int,
                    metavar="N", help="Save ttyrec every N episodes.")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.0006,
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "tanh", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.00048,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")
# yapf: enable


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def act(
    flags,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        env = create_env(flags, seed, False)
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state, _ = model(env_output, agent_state)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout.
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state, _ = model(env_output, agent_state)

                timings.time("model")

                env_output = env.step(agent_output["action"])

                timings.time("step")

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )
    timings.time("device")
    return batch, initial_agent_state


def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        learner_outputs, unused_state, _ = model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "tanh":
            clipped_rewards = torch.tanh(rewards / 100)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())
        return stats


def predictor_learn(
    flags,
    model,
    batch,
    optimizer,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        prediction_outputs, obs_embedding, oa_embedding = model(batch)

        rewards = batch["reward"]
        reward_prediction_loss = (prediction_outputs["reward"] - rewards / 10).square().mean()

        contrast_loss = 0
        cnt = 0

        T, B = rewards.shape
        for i in range(B):
            embeds = []
            for j in range(T):
                if abs(rewards[j, i].item()) > 0.5:
                    embeds.append(oa_embedding[j, i])
                if batch["done"][j, i].item() or j == T - 1:
                    L = len(embeds)
                    if L > 1:
                        _embeds = torch.stack(embeds, 0)
                        in_dis = (_embeds.unsqueeze(0) - _embeds.unsqueeze(1)).square().sum(-1)
                        d = torch.exp(-in_dis / in_dis.max() * 2)
                        c = torch.det(d)
                        contrast_loss -= c
                        cnt += 1
                    embeds = []

        if cnt > 0:
            contrast_loss /= cnt
        total_loss = reward_prediction_loss + contrast_loss

        stats = {
            "reward_prediction_loss": reward_prediction_loss.item(),
            "contrast_loss": contrast_loss.item() if cnt > 0 else 0,
            "predictor_total_loss": total_loss.item(),
        }

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()

        return stats


def create_buffers(flags, observation_space, num_actions, num_overlapping_steps=1):
    size = (flags.unroll_length + num_overlapping_steps,)

    # Get specimens to infer shapes and dtypes.
    samples = {k: torch.from_numpy(v) for k, v in observation_space.sample().items()}

    specs = {
        key: dict(size=size + sample.shape, dtype=sample.dtype)
        for key, sample in samples.items()
    }
    specs.update(
        reward=dict(size=size, dtype=torch.float32),
        done=dict(size=size, dtype=torch.bool),
        episode_return=dict(size=size, dtype=torch.float32),
        episode_step=dict(size=size, dtype=torch.int32),
        policy_logits=dict(size=size + (num_actions,), dtype=torch.float32),
        baseline=dict(size=size, dtype=torch.float32),
        last_action=dict(size=size, dtype=torch.int64),
        action=dict(size=size, dtype=torch.int64),
    )
    buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    torch.set_num_threads(30)

    load = flags.loaddir is not None
    if load:
        loadpath = os.path.expandvars(
            os.path.expanduser("%s/%s" % (flags.loaddir, "model.tar"))
        )
        load_checkpoint_cpu = torch.load(loadpath, map_location="cpu")
        load_checkpoint = torch.load(loadpath, map_location=flags.device)

    def show_state_dict(state_dict):
        for item in state_dict:
            print(item)
        print()

    env = create_env(flags, None, False, savedir=flags.savedir, save_ttyrec_every=flags.save_ttyrec_every)

    model = net.get_policy(flags.env, env.observation_space, env.action_space.n, flags.use_lstm)

    # show_state_dict(model.state_dict())

    if load and "model_state_dict" in load_checkpoint_cpu:
        # show_state_dict(load_checkpoint_cpu["model_state_dict"])
        model.load_state_dict(load_checkpoint_cpu["model_state_dict"])

    buffers = create_buffers(flags, env.observation_space, model.num_actions)

    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                i,
                free_queue,
                full_queue,
                model,
                buffers,
                initial_agent_state_buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)

    learner_model = net.get_policy(flags.env,
        env.observation_space, env.action_space.n, flags.use_lstm
    ).to(device=flags.device)

    if load and "model_state_dict" in load_checkpoint:
        learner_model.load_state_dict(load_checkpoint["model_state_dict"])

    prediction_model = net.get_prediction_net(flags.env, env.observation_space, env.action_space.n).to(device=flags.device)

    if load and "prediction_model_state_dict" in load_checkpoint:
        prediction_model.load_state_dict(load_checkpoint["prediction_model_state_dict"])
        
    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    if load and "optimizer_state_dict" in load_checkpoint:
        optimizer.load_state_dict(load_checkpoint["optimizer_state_dict"])

    prediction_optimizer = torch.optim.Adam(
        prediction_model.parameters(),
        lr=2.5e-4,
        eps=1e-5,
    )

    if load and "prediction_optimizer_state_dict" in load_checkpoint:
        prediction_optimizer.load_state_dict(load_checkpoint["prediction_optimizer_state_dict"])


    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if load and "scheduler_state_dict" in load_checkpoint:
        scheduler.load_state_dict(load_checkpoint["scheduler_state_dict"])
    # print("111", flush=True)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
        "predictor_total_loss",
        "reward_prediction_loss",
        "contrast_loss"
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            logger.info(f"{i}, {step}")
            timings.reset()
            batch, agent_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
                timings,
            )
            if not flags.no_actor_learning:
                learner_stats = learn(
                    flags, model, learner_model, batch, agent_state, optimizer, scheduler
                )
            else:
                learner_stats = {}
            if not flags.no_predictor_learning:
                predictor_stats = predictor_learn(
                    flags, prediction_model, batch, prediction_optimizer
                )
            else:
                predictor_stats = {}
            stats = {**learner_stats, **predictor_stats}
            # print(predictor_stats, stats, flush=True)

            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys if k in stats})
                plogger.log(to_log)
                step += T * B

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "prediction_model_state_dict": prediction_model.state_dict(),
                "prediction_optimizer_state_dict": prediction_optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        sps_deque = deque(maxlen=20)
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > 5 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            sps_deque.append(sps)
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            predictor_total_loss = stats.get("predictor_total_loss", float("inf"))
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. Predictor loss %f. %sStats:\n%s",
                step,
                sum(sps_deque) / len(sps_deque),
                total_loss,
                predictor_total_loss,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()


def test(flags, num_episodes: int = 10, save_data: bool = True):
    if flags.xpid is None:
        checkpointpath = "./latest/model.tar"
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
        )

    if flags.test_episodes is not None:
        num_episodes = flags.test_episodes

    env = create_env(flags, None, True, save_ttyrec_every=flags.save_ttyrec_every)
    model = net.get_policy(flags.env, env.observation_space, env.action_space.n, flags.use_lstm)
    # model.eval()
    prediction_model = net.get_prediction_net(flags.env, env.observation_space, env.action_space.n)
    prediction_model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    prediction_model.load_state_dict(checkpoint["prediction_model_state_dict"])
    # print(prediction_model.state_dict())

    observation = env.initial()
    returns = []

    achievement_counts = defaultdict(int)

    if flags.env.startswith('NetHack'):
        obs_keys = ['glyphs', 'blstats', 'chars']
    else:
        obs_keys = ['frame']

    keys = ['action', 'policy_logits', 'hidden', 'baseline', 'reward', 'unlocked', 'reward_prediction', 'obs_hidden', 'oa_hidden', *obs_keys]
    data_cur = { key: [] for key in keys }
    for key in obs_keys:
        data_cur[key].append(observation[key])
    episode_data = { key: [] for key in keys }

    while len(returns) < num_episodes:
        if flags.mode == "test_render":
            env.gym_env.render()
        agent_outputs = model(observation)
        policy_outputs, _, policy_hidden = agent_outputs
        last_obs = { key: observation[key].clone() for key in obs_keys }
        observation = env.step(policy_outputs["action"])

        preidctor_outputs, obs_hidden, oa_hidden = prediction_model({
            "action": policy_outputs["action"],
            ** { key: observation[key] for key in obs_keys },
            ** { f'last_{key}': last_obs[key] for key in obs_keys },
        })

        if save_data:
            data_cur['action'].append(policy_outputs["action"])
            data_cur['policy_logits'].append(policy_outputs["policy_logits"])
            data_cur['baseline'].append(policy_outputs["baseline"])
            data_cur['hidden'].append(policy_hidden)
            data_cur['reward'].append(observation["reward"])
            data_cur['reward_prediction'].append(preidctor_outputs["reward"])
            data_cur['obs_hidden'].append(obs_hidden)
            data_cur['oa_hidden'].append(oa_hidden)
            if "unlocked" in observation["info"]:
                data_cur['unlocked'].append(list(observation["info"]["unlocked"]))

        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
            if 'achievements' in observation['info']:
                for a, c in observation['info']['achievements'].items():
                    if c > 0:
                        achievement_counts[a] += 1
            if save_data:
                for key, value in data_cur.items():
                    if len(value) > 0:
                        if isinstance(value[0], torch.Tensor):
                            episode_data[key].append(torch.concat(value, 0).squeeze(1).detach().numpy())
                        else:
                            episode_data[key].append(value)
                data_cur = { key: [] for key in keys }
                for key in obs_keys:
                    data_cur[key].append(observation[key].clone())
        elif save_data:
            for key in obs_keys:
                data_cur[key].append(observation[key].clone())

    env.close()
    logging.info(
        "Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns)
    )

    logging.info("Achivement counts:")
    for a, c in sorted(achievement_counts.items(), key=lambda x: x[1]):
        logging.info(f"{a}: {c} ({c / len(returns):.2%})")

    if save_data:
        joblib.dump(episode_data, os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "test.data"))
        ))


def create_env(flags, seed, append_info, *args, **kwargs):
    if flags.env == 'crafter':
        env = atari_wrappers.wrap_pytorch(
            atari_wrappers.make_crafter()
        )
        if seed is not None:
            env.seed(seed)
        return environment.Environment(env, append_info)
    elif flags.env.startswith('NetHack'):
        env = gym.make(flags.env, observation_keys=("glyphs", "blstats", "chars"), *args, **kwargs)
        if seed is not None:
            env.seed(seed)
        return environment.NetHackEnvironment(env, append_info)
    else:
        env = atari_wrappers.wrap_pytorch(
            atari_wrappers.wrap_deepmind(
                atari_wrappers.make_atari(flags.env),
                clip_rewards=False,
                frame_stack=True,
                scale=False,
            )
        )
        if seed is not None:
            env.seed(seed)
        return environment.Environment(env, append_info)


def main(flags):
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
