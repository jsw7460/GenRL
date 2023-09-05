import json
import pickle

import gymnasium as gym
import h5py
import numpy as np
from minari.serialization import serialize_space

d4rl_filename = "/home/jsw7460/kitchen_skill_appended_Total.pkl"


onehot_skills_mapping = {
    'bottom burner': 0,
    'top burner': 1,
    'light switch': 2,
    'slide cabinet': 3,
    'hinge cabinet': 4,
    'microwave': 5,
    'kettle': 6,
}


with open(d4rl_filename, "rb") as fp:
    dataset = pickle.load(fp)


# exit()
observations = np.squeeze(dataset["observations"])
actions = np.squeeze(dataset["actions"])
rewards = np.squeeze(dataset["rewards"])
skills_str = dataset["skills"]
skills = np.array([onehot_skills_mapping[sk] for sk in skills_str], dtype="i4")
skills_done = np.squeeze(dataset["skill_done"])
terminations = np.squeeze(dataset["terminals"])
truncations = np.squeeze(dataset["terminals"])

f = h5py.File("/home/jsw7460/Kitchen-v0", "w")

obs_dim = observations.shape[-1]
act_dim = actions.shape[-1]
observation_space = gym.spaces.Box(low=-np.infty, high=np.infty, shape=(obs_dim,))
action_space = gym.spaces.Box(low=np.min(actions, axis=0), high=np.max(actions, axis=0), shape=(act_dim,))

n_episode = 0
ep_start = 0

for t in range(len(observations)):
    if terminations[t] or truncations[t]:
        episode = f.create_group(f"episode_{n_episode}")

        ep_obs = observations[ep_start: t + 1]
        ep_act = actions[ep_start: t + 1]
        ep_rew = rewards[ep_start: t + 1]
        ep_terminations = terminations[ep_start: t + 1]
        ep_truncations = truncations[ep_start: t + 1]

        ep_skills = skills[ep_start: t + 1][..., np.newaxis]
        ep_skills_done = skills_done[ep_start: t + 1]

        episode.create_dataset(
            name="observations",
            shape=ep_obs.shape,
            dtype=ep_obs.dtype,
            data=ep_obs
        )

        episode.create_dataset(
            name="actions",
            shape=ep_act.shape,
            dtype=ep_act.dtype,
            data=ep_act
        )

        episode.create_dataset(
            name="rewards",
            shape=ep_rew.shape,
            dtype=ep_rew.dtype,
            data=ep_rew,
        )

        episode.create_dataset(
            name="terminations",
            shape=ep_terminations.shape,
            dtype=ep_terminations.dtype,
            data=ep_terminations
        )

        episode.create_dataset(
            name="truncations",
            shape=ep_truncations.shape,
            dtype=ep_truncations.dtype,
            data=ep_truncations
        )

        episode.create_dataset(
            name="sem_skills",
            shape=ep_skills.shape,
            dtype=ep_skills.dtype,
            data=ep_skills
        )
        print(ep_skills.shape)

        episode.create_dataset(
            name="sem_skills_done",
            shape=ep_skills_done.shape,
            dtype=ep_skills_done.dtype,
            data=ep_skills_done
        )

        episode.attrs["id"] = n_episode
        episode.attrs["seed"] = 0
        episode.attrs["total_steps"] = len(ep_obs)

        ep_start = t + 1
        n_episode += 1

env_spec = {"id": "halfcheetah-expert-v0", "additional_wrappers": []}
f.attrs["total_episodes"] = n_episode
f.attrs["total_steps"] = len(observations)
f.attrs["observation_space"] = serialize_space(observation_space)
f.attrs["action_space"] = serialize_space(action_space)
f.attrs["dataset_id"] = "kitchen-expert-v0"
f.attrs["env_spec"] = json.dumps(env_spec)
f.attrs["minari_version"] = ">0.3.1"

print(f.keys())
print(f.attrs.keys())
print(f.attrs["minari_version"])
