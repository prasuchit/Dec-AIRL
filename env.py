#!/usr/local/bin/python3

# The MIT License (MIT)

# Copyright (c) 2022 Prasanth Suresh and Yikang Gui

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import gym

gym.logger.set_level(40)


def make_env(env_id):
    return NormalizedEnv(gym.make(env_id))


class NormalizedEnv(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps

        if hasattr(env.action_space, 'high'):
            self.scale = env.action_space.high
            self.action_space.high /= self.scale
            self.action_space.low /= self.scale
            self.action_discrete = False
        else:
            self.action_discrete = True
            self.scale = 1

    def step(self, action):
        if self.action_discrete:
            return self.env.step(int(action.item()))
        else:
            return self.env.step(action * self.scale)
