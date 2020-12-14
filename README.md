# cs744_assignments
CS744-Assignments

## Running project Python code in GCP
To run the Python scripts found in `/project/` on GCP, follow these steps:
1. Create a general purpose, N1 class, standard-1 (not shared core) instance in GCP.
1. Under “CPU platform and GPU”, add your desired GPU.
1. Change boot disk from default 10GB to 20GB (CUDA requires a ton of space).
1. (Optional) Under “Management, security, disks, …” set Availability Policy -> Preemptibility to “On” (this cuts cost to ~30%, but means the instance won’t run longer than 24 hours).
1. Click the `SSH` button to connect to the instance via SSH in a browser window. 🤯
1. Run `sudo apt install -y git` to be able to clone our git repo.
1. Run `git clone https://github.com/kamikazekartik/cs744_assignments.git` to pull down the code (log in with your GitHub credentials).
1. Run `cd cs744_assignments/project/` to get into the project directory.
1. Run `chmod +x setup_init.sh && chmod +x setup_finish.sh` to make the setup scripts executable.
1. Run `./setup_init.sh`. When complete, this will terminate your SSH session for reboot (to install a kernel upgrade); click the `SSH` button on your instance to reconnect.
1. Run `cd cs744_assignments/project/` to get into the project directory again, and run `./setup_finish.sh` to complete setup (this takes a while). 

When finished, you can run the code using the bash script `run_main.sh`.
Typical usage is going to look like `nohup bash run_main.sh &`. NOTE: The outputs will be written to the log file which is specified at the bottom of `run_main.sh`.

## Running EB Ticket Code
For setting up the `early_bird` enviornment, run the following scripts after previous setup.
1. Run `bash ./setup_conda.sh` to build miniconda3 in your home directory. When prompted, respond with `Enter` or `yes`
1. Run `bash ./setup_env.sh` to setup and activate the `early_bird` environment.

You can always exit conda completely by running `conda deactivate` until the conda prompt has exited. To retrun to the EarlyBird environment, you can run `~/miniconda3/bin/activate` followed by `source activate early_bird`.

The scripts for building and running on tickets are in the `/project/Early-Bird-Tickets/` directory. From `/project/Early-Bird-Tickets/`, run the command `bash ./scripts/vgg-fp/search.sh` to find a ticket and `bash ./scripts/vgg-fp/pruned.sh` to prune it. Similarly, to recover the low-precision tickets, you can run `bash ./scripts/vgg-lp/search.sh` to get the low precision ticket and `bash ./scripts/vgg-lp/pruned.sh` to prune it.

These tickets can be called form the `/project/` directory using the scripts `bash run_main_eb.sh` and `bash run_main_eb_lp.sh`.

## PreResNet
To run PreResnet, you can do one of two things:

1. Run 'non-pruned', by running `run_preresnet.sh` file while omitting the `--pruned-model-path` argument. This will train from scratch like any normal model.
1. Run on a 'pruned' model. This can be done by first recovering the EB ticket, and then running the `run_preresnet.sh` script with a path to the EB ticket model .pth.tar.

If you run without EB ticket, it should just work out of the box as all other models. With EB ticket, you can generate tickets with the following familiar steps:

1. Activate the early_bird environment (optional?)
1. In `/early_bird_tickets`, run `bash ./scripts/preresnet/search.sh`. Note - this script will run for a fixed number of iterations (a long time), and will save out `/baseline/resnet-nolp-cifar10/EB-30.pth.tar` (model for 30 percent pruned) and `/baseline/resnet-nolp-cifar10/EB-50.pth.tar` (model for 50 percent pruned) while running. Once the desired networks have been created, you can kill the script safely.
1. Run `bash ./scripts/preresnet/prune30.sh` and `bash./scripts/preresnet/prune50.sh` to prune and save out the pruned models.

From here, you should be able to load these models into your `run_preresnet.sh` model.

In terms of parameters, the EB Ticket authors noted the following parameters for training:

Training takes 160 epochs in
total and the batch size of 256; the initial learning rate is set to 0.1, and is divided by 10 at the 80th
and 120th epochs, respectively; the SGD solver is adopted with a momentum of 0.9 and a weight
decay of 10−4