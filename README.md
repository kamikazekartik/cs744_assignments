# cs744_assignments
CS744-Assignments

## Running project Python code in GCP
To run the Python scripts found in `/project/` on GCP, follow these steps:
1. Create a general purpose, N1 class, standard-1 (not shared core) instance in GCP.
1. Under “CPU platform and GPU”, add your desired GPU.
1. Change boot disk from 10-20 GB (CUDA requires a ton of space).
1. (Optional) Under “Management, security, disks, …” set Availability Policy -> Preemptibility to “On” (this cuts cost to ~30%, but means the instance won’t run longer than 24 hours).
1. Click the `SSH` button to connect to the instance via SSH in a browser window. 🤯
1. Run `sudo apt install -y git` to be able to clone our git repo.
1. Run `git clone https://github.com/kamikazekartik/cs744_assignments.git` to pull down the code (log in with your GitHub credentials).
1. Run `cd cs744_assignments.git/project/` to get into the project directory.
1. Run `chmod +x setup_init.sh && chmod +x setup_finish.sh` to make the setup scripts executable.
1. Run `./setup_init.sh`. When complete, this will terminate you SSH session for reboot (to install a kernel upgrade); click the `SSH` button on your instance to reconnect.
1. Run `cd cs744_assignments.git/project/` to get into the project directory again, and run `./setup_finish.sh` to complete setup (this takes a while)
When finished, you can run python scripts with `python3 main.py` or similar.
