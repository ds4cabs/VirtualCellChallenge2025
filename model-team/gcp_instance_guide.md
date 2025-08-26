# A Guide to Training State Models on Google Cloud Platform

This guide walks through the process of setting up a Google Cloud Platform (GCP) virtual machine to train our state models. We'll cover everything from the initial instance configuration to connecting to the machine, transferring files, and running the training script.

## 1. GCP Instance Configuration

To handle the computational demands of training our models, we need a GCP instance with specific hardware. Here's a summary of a recommended configuration:

*   **Machine Type:** `g2-standard-8` (8 vCPUs, 32 GB Memory)
*   **GPU:** 1 x NVIDIA L4
*   **Boot Disk:** 250 GB SSD persistent disk
*   **Image:** `c0-deeplearning-common-cu113-v20241118-debian-11` (pre-configured with CUDA drivers)
*   **Location:** `us-central1-a` (or your preferred region)

This setup provides a powerful GPU and sufficient memory for our training needs. The deep learning-optimized image comes pre-installed with many useful libraries and drivers.

## 2. Connecting to the Instance

Google Cloud's `gcloud` command-line tool makes it easy to connect to your instance via SSH. Once your instance is running, open your terminal and use the following command, replacing `<your-instance-name>` with your actual instance name:

```bash
gcloud compute ssh <your-instance-name>
```

This command will securely connect you to the instance's terminal. All subsequent commands are to be run on the GCP instance unless otherwise specified.

## 3. Preparing Directories and Transferring Files

This workflow assumes you are running a local version of the `state` source code that you want to test on a powerful cloud machine.

First, create the necessary directories on the GCP instance. Make sure to replace `<your-username>` with your username on the instance.

```bash
# Create directories for the source code, training scripts, and data
mkdir -p /home/<your-username>/state/
mkdir -p /home/<your-username>/prepare_instance/
mkdir -p /home/<your-username>/SE600M_embedding/
```

Next, from your **local machine's terminal**, use the `gcloud compute scp` command to upload your code. This command recursively copies your local directories to the directories you just created on the instance.

Replace `<your-local-path>`, `<your-instance-name>`, `<your-username>`, and `<your-zone>` with your specific information.

```bash
# (Run from your local machine)
# Upload the state source code
gcloud compute scp --recurse <your-local-path>/state/ <your-instance-name>:/home/<your-username>/state/ --zone <your-zone>

# Upload the prepare_instance directory
gcloud compute scp --recurse <your-local-path>/prepare_instance/ <your-instance-name>:/home/<your-username>/prepare_instance --zone <your-zone>
```

## 4. Setting up the Environment

Now that your code is on the instance, you need to install its dependencies and download the required data.

```bash
# (Run on the GCP instance)
# Install necessary Python packages
pip install huggingface_hub uv

# Navigate to your source code directory
cd /home/<your-username>/state

# Use uv to install the dependencies defined in the project
# This makes the `state` command available
uv sync

# Download cell embedding data from Google Cloud Storage
# Replace <your-gcs-bucket> with the name of your GCS bucket
gsutil rsync -r gs://<your-gcs-bucket>/ /home/<your-username>/SE600M_embedding/

# Alternatively, data is available at https://huggingface.co/datasets/VirtualCell2025/SE600M-embedding
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='VirtualCell2025/SE600M-embedding', repo_type='dataset', local_dir='/home/<your-username>/SE600M_embedding')"

# Download ESM2 protein embeddings from Hugging Face
python -c "from huggingface_hub import hf_hub_download;hf_hub_download(repo_id='arcinstitute/SE-600M', filename='protein_embeddings.pt',local_dir='/home/<your-username>/SE600M_embedding')"

# (Optional) Configure git and Hugging Face login if you plan to upload results
# The following steps are only necessary if you want to push models or datasets
# to the Hugging Face Hub. Downloading public files does not require a login.
git config --global credential.helper store
huggingface-cli login
```

## 5. Understanding the `prepare_instance` Directory

The `prepare_instance` directory contains the core scripts and configurations for our training job. While you transfer the whole directory, three files are especially important:

*   **`run_tx_training.sh`**: This is the main executable script that launches the training process. It calls the `state` package with all the necessary parameters for the model, data, and training configuration. Here is the full command:

    ```bash
    uv run state tx train \
     data.kwargs.toml_config_path="/home/<your-username>/prepare_instance/starter.toml" \
     data.kwargs.num_workers=4 \
     data.kwargs.embed_key="X_state" \
     data.kwargs.batch_col="batch_var" \
     data.kwargs.pert_col="target_gene" \
     data.kwargs.cell_type_key="cell_type" \
     data.kwargs.control_pert="non-targeting" \
     data.kwargs.perturbation_features_file="/home/<your-username>/SE600M_embedding/protein_embeddings.pt" \
     training.batch_size=24 \
     training.max_steps=40000 \
     +training.accumulate_grad_batches=8 \
     training.ckpt_every_n_steps=4000 \
     model=tahoe_sm \
     use_wandb=false \
     output_dir="/home/<your-username>/competition" \
     name="test"
    ```

*   **`starter.toml`**: This TOML file is crucial for defining your dataset. It specifies the paths to your data files and how the data should be split for training, validation, and testing (e.g., defining which cell types are used for zeroshot evaluation). The `run_tx_training.sh` script directly references this file.

*   **`train.log`**: This file is generated when you run the training script with `nohup`. It captures all the output from the training process, including progress, performance metrics, and any errors. It's essential for monitoring the job and debugging any issues.

## 6. Running the Training

With the environment set up and the necessary files in place, you can now start the training process.

First, make the training script executable:

```bash
chmod +x /home/<your-username>/prepare_instance/run_tx_training.sh
```

To run the training in the background from the `state` directory and save the output to a log file, use `nohup`:

```bash
cd /home/<your-username>/state
nohup ../prepare_instance/run_tx_training.sh > ../prepare_instance/train.log 2>&1 &
```

You can monitor the training progress by tailing the `train.log` file. Once the training is complete, you can use `gcloud compute scp` (from your local machine) to download the results and logs for analysis.
