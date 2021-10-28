# DiagnosisCorrection02456
Project for 02456 Deep Learning

# Setting up on HPC

First, call 

`s174483@login1.gbar.dtu.dk` 

Clone the repository

`git clone git@github.com:vstenby/DiagnosisCorrection02456.git`

Go into the folder

`cd DiagnosisCorrection02456/`

Load the Python3 module

`module load python3`

Make a virtual environment in the folder

`virtualenv venv` 

Activate the virtual environment

`source venv/bin/activate`

Install jupyter

`python -m pip install jupyter`

Install jupyterlab

`python -m pip install jupyterlab`

Open jupyterlab by typing

``jupyter lab --no-browser``

Find the port (e.g. localhost:8888) and open a new terminal window on your machine.

Forward the port in the new terminal window by typing (here, the port used was 8888)

``ssh s174483@login1.gbar.dtu.dk -NL 8888:localhost:8888` 

Once you have entered in your password, you can log in by going to e.g. Google Chrome and open the link in your first terminal (or by going to localhost:8888 in your browser).

Note, if you want to install more packages, you can in jupyter lab e.g. write `!python -m pip install numpy`

## Running stuff on GPU

Once you have JupyterLab open, open a new terminal here by pressing File -> New -> Terminal and enter one of the three commands:
* `voltash`
* `sxm2sh`
* `a100sh`

This will launch an interactive session on a GPU, where you can execute e.g. the `train.py` script (after activating your virtual environment). 



