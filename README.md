# Pytorchasaurus Rex
Pytorch implementation of DQN, 1-step SARSA, REINFORCE, A2C, 1-step actor critic on Google Chrome dinosaur  
  
![dinosaur running](./dino.gif)


## Installation
We recommend creating a virtualenv before installing the required packages. See [virtualenv](https://virtualenv.pypa.io/en/stable/) or [virtualenv-wrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) on how to do so.

The dependencies can be easly installed using pip.
```sh
$ optional: open the virtualenv
$ pip install -r requirements.txt
```

## Getting started

### Webserver for running the javascript T-rex game

A simple webserver is required to run the T-rex javascript game.
The easiest way to achieve this is by using python's Simple HTTP Server module.  

Open a new terminal and navigate to `TF-Rex/game`, then run the following command
```sh
$ cd /path/to/project/TF-Rex/game
$ python3 -m http.server
```
The game is now accessable on your localhost `127.0.0.1:8000`.
This approach was tested for Chrome and Mozilla Firefox.

### Tf-Rex

```sh
$ python test.py
```
This command will restore the pretrained model, stored in `tf-rex/results` and play the T-rex game, default is one step critic (our best model)

IMPORTANT: The browser needs to connect with the python side. Therefore, refresh the browser after firing the training/testing command.

Training a new model can be done as follow depending on the model that you want to train
```sh
$ python dqn.py
$ python a2c.py
$ python sarsa.py
$ python reinforce.py
$ python one_step_actor_critic.py
```

## References
[1] [The original TF-Rex](https://vdutor.github.io/blog/2018/05/07/TF-rex.html)  
[2] [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
