# Dec-AIRL

Author: Prasanth Sengadu Suresh and Yikang Gui.

Owned By: THINC Lab, Department of Computer Science,
          University of Georgia.

Currently Managed By: Prasanth Sengadu Suresh(ps32611@uga.edu) and Yikang Gui(Yikang.Gui@uga.edu).

This package extends [Adversarial Inverse Reinforcement Learning](https://arxiv.org/pdf/1710.11248.pdf) to use joint expert demonstrations to learn decentralized policies for the actors to work collaboratively. This is implemented on a real world use-inspired produce sorting domain where a Human expert sorts produce alongside a UR3e Robot.

The code was extended [from this base package](https://github.com/ku2482/gail-airl-ppo.pytorch).

## The following instructions were tested on Ubuntu 18.04 and 20.04. Please use only the main branch, the other branches are either deprecated or are still being developed.

The following are the steps to be followed to get this package working:

  You need a GPU capable of running Pytorch and Cuda to run this code efficiently. This was tested with an Nvidia Quadro Rtx 4000 and GeForce GTX 1080.

  0.) Clone the package:
        git clone https://github.com/prasuchit/Dec-AIRL.git
  
  1.) Install Dependencies
  
   [Anaconda install](https://docs.anaconda.com/anaconda/install/linux/)
      
   [Mujoco 200 install](https://brucknem.github.io/posts/setup-mujoco-py-and-robosuite/)

   [Multi-agent Gym](https://github.com/prasuchit/ma-gym.git)

   [Assistive Gym](https://github.com/prasuchit/assistive-gym)
   
   Do(optional):     
   
          sudo apt-get update && sudo apt-get upgrade && sudo apt-get dist-upgrade
   
  2.) Cd into the package folder and create a new conda env using the environment yml file:
   
        conda env create --file environment.yml
     
   - You may run into dependency issues here regarding pytorch, mujoco or the like. From our experience, install instructions can become outdated for these libraries quite soon. Google and StackOverflow are your best friends at this point. But feel free to raise any issues, we'll try to address it the best we can. 

   - *** From my experience, since assistive gym and ma_gym use different gym versions, they cause a lot of conflicts. I would suggest using seperate conda environments.
      
  3.) Assuming you've crossed all these steps fully, activate the conda env:

        conda activate dec_airl

  3.1) Make sure you've installed ma_gym and assistive_gym correctly. Following the instructions on their Readme should work fine.
    
        cd Dec-AIRL && mkdir -p buffers models models_airl

  4.) The following commands can be used to train an expert, record trajectories for DecHuRoSorting-v0 and run Multi-agent AIRL HuRo-TWIRL:

        python3 scripts/rl.py --env ma_gym:DecHuRoSorting-v0

        python3 scripts/record.py --env ma_gym:DecHuRoSorting-v0

        python3 scripts/irl.py --env ma_gym:DecHuRoSorting-v0

The same commands can be used with modified env argument to run assistive gym envs. Refer to the code for the right spelling and format.
There are several commandline arguments that can be adjusted. All arguments are not listed here. 

If the learning accuracy isn't improving, try increasing/decreasing the size of the discriminator hidden nodes. AIRL works on a delicate balance between generator and discriminator, if one overpowers the other, learning suffers. 
        
### Please raise your issues or questions here, we'll address it as soon as possible. We may not answer email queries.
