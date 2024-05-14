# Description of the algorithm
I have decided to use NEAT.
Our fitness evaluation function (eval_genome) calculates the mean fitness over the multiple runs.
In my experiments, nruns = 5 worked quite well.
This helps assess the performance stability of the agents. Very crucial for enviroments with high
variability. While the episode has not terminated, or truncated(episode length exceeded) we pass
the observations through the activation function, selecting the maximum response and then applying
the action.
# Configs
I have achieved reasonable results with Acrobot, CartPole, LunarLander, and Mountain Car Countinous 
enviroments. The parameters in the config files more or less repeat, with the exception of number
of inputs/outputs, and fitness threshold those depend on the enviroment. 
Only for the cartpole enviroment, i opted for mean fitness criterion as it results in more 
generalizable solutions.

For all of them I set 2 hidden neurons to allow more complex nonlinear relationships 
(such as the relationship between the position of the cart and the angle of the pole in Cartpole). 
For Box2d enviroments this was especially crucial, as it would struggle to learn all of the actions 
to keep itself stable otherwise. 
Setting the intial_connection parameter to partial_direct 0.5. By starting with only 50% of possible 
connections, the initial networks are simpler. This prevents overwhelming the early stages of evolution 
with overly complex structures that may be difficult to optimize. 

Clamped activation function works well for most of the enviroments. Sin did work slightly better 
for the acrobot enviroment, however it was negligible and resulted in longer evolution.



# Usage

To evaluate:
```
python sweet_neat.py --env <gymnasium-env-name> --conf configs/<config_name> --eval
```

To evolve and run the best winner:
```
python sweet_neat.py --env <gymnasium-env-name> --conf configs/<config_name> --evolve -- show
```
For example:
```
python sweet_neat.py --env CartPole-v1 --conf configs/config-cartpole --evolve -- show
```  
