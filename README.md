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