#! /bin/bash
set -e
set -x

source /fsx/mengchiy/eric_fork/venv/bin/activate

# sudo dpkg -r aws-neuronx-runtime-lib-debug
# sudo dpkg -i /fsx/mengchiy/aws-neuronx-dkms_2.x.4290.0_amd64.deb
sudo dpkg -i /fsx/mengchiy/aws-neuronx-collectives-2.x.25541.0-6b8a083a4.deb
sudo dpkg -i /fsx/mengchiy/aws-neuronx-runtime-lib-2.x.23913.0-b413f6965.deb

# sudo apt-get install aws-neuronx-tools=2.* -y


# sudo apt-get -f install -y
# sudo apt-get install -y google-perftools
# sudo apt -y --fix-broken install

# echo "after fix broken"
# sudo apt-get install -y google-perftools
# echo "after install"

# Neuron env vars for distributed training based on SLURM
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
num_nodes=$(echo "$nodes" | wc -l)
devices_per_node=64
MASTER_ADDR=$(echo "$nodes" | head -n 1)
MASTER_PORT=41000
JAX_COORDINATOR_PORT=41001
export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"
export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '%s,' $(seq 1 $num_nodes | xargs -I {} echo $devices_per_node) | sed 's/,$//')
export NEURON_PJRT_PROCESS_INDEX=$SLURM_NODEID

cd /fsx/mengchiy/eric_fork/axlearn
pip list

# Editable paths
ARTIFACTS_PATH="/fsx/mengchiy/eric_fork/artifacts"
# TIMESTAMP=$(date +"%y%m%d%H%M%S")
TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/$SLURM_JOB_ID"
# TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/9111"
mkdir -p "$TEST_ARTIFACTS_PATH"

NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump

python -c "import sys; print(sys.path)"

source flag_list.sh $NEURON_DUMP_PATH $HLO_DUMP_PATH

# LIBTCMALLOC=$(find /usr/lib/x86_64-linux-gnu -name "libtcmalloc.so.*" | sort -V | tail -n 1)

# if [ -n "$LIBTCMALLOC" ]; then
#     # Create a symbolic link to the found libtcmalloc version
#     sudo ln -sf "$LIBTCMALLOC" /usr/lib/libtcmalloc.so
#     echo "Symbolic link created: /usr/lib/libtcmalloc.so -> $LIBTCMALLOC"

#     # Export LD_PRELOAD
#     export LD_PRELOAD=/usr/lib/libtcmalloc.so
#     echo "LD_PRELOAD set to: $LD_PRELOAD"
# else
#     echo "Error: libtcmalloc.so not found"
#     exit 1
# fi


MODULE="text.gpt.c4_trainer"
CONFIG=$1 # which config to use is passed into script as argument
BACKEND="neuron"
MESH="neuron-trn2n.48xlarge-64"

# trainer related directory #FIXME:
OUTPUT_DIR="${TEST_ARTIFACTS_PATH}/axlearn_out"
mkdir -p ${OUTPUT_DIR}
DATA_DIR="gs://axlearn-public/tensorflow_datasets"

echo "Listing apt dependencies"
apt list --installed | grep neuron
echo "Listing pip dependencies"
pip list | grep neuron
echo "Done listing dependencies"

which python

echo "Listing hostname"
hostname

unset JAX_COMPILATION_CACHE_DIR

echo "printing env"
printenv | grep "NEURON"
printenv | grep "LD"

# Function to write zero to all peak files
write_zero_to_peaks() {
    local base_path="/sys/devices/virtual/neuron_device/neuron0/neuron_core0/stats/memory_usage/device_mem"
    local categories=("collectives" "constants" "dma_rings" "driver_memory" "model_code" "model_shared_scratchpad" "nonshared_scratchpad" "notifications" "runtime_memory" "tensors" "uncategorized")

    for category in "${categories[@]}"; do
        sudo bash -c "echo 0 > ${base_path}/${category}/peak"
    done
    sudo bash -c "echo 0 > ${base_path}/peak"
}

# Function to read and print summary of peak memory
read_peak_memory() {
    local base_path="/sys/devices/virtual/neuron_device/neuron0/neuron_core0/stats/memory_usage/device_mem"
    local categories=("collectives" "constants" "dma_rings" "driver_memory" "model_code" "model_shared_scratchpad" "nonshared_scratchpad" "notifications" "runtime_memory" "tensors" "uncategorized")

    # Function to convert bytes to GB with 3 decimal places
    bytes_to_gb() {
        echo "scale=3; $1 / 1024 / 1024 / 1024" | bc
    }

    echo "Peak Memory Summary:"
    for category in "${categories[@]}"; do
        local peak=$(sudo cat "${base_path}/${category}/peak")
        local peak_gb=$(bytes_to_gb $peak)
        printf "%s: %s GB\n" "${category}" "${peak_gb}"
    done

    local total_peak=$(sudo cat "${base_path}/peak")
    local total_peak_gb=$(bytes_to_gb $total_peak)
    printf "Total Peak: %s GB\n" "${total_peak_gb}"
}

# export JAX_PLATFORMS='cpu'
# BACKEND="cpu"
# export XLA_FLAGS="${XLA_FLAGS} --xla_force_host_platform_device_count=64"

# write_zero_to_peaks

# if git apply --reverse --check neuron_custom_hook.patch; then
#     echo "Patch already there."
# else
#     git apply neuron_custom_hook.patch
#     echo "Patch applied successfully."
# fi

python -m axlearn.common.launch_trainer_main \
    --module=${MODULE} \
    --config=${CONFIG} \
    --trainer_dir=${OUTPUT_DIR} \
    --data_dir=${DATA_DIR} \
    --jax_backend=${BACKEND} \
    --mesh_selector=${MESH} \
    --distributed_coordinator=$MASTER_ADDR:$JAX_COORDINATOR_PORT --num_processes=$num_nodes \
    --process_id=$NEURON_PJRT_PROCESS_INDEX
# pytest /fsx/mengchiy/fs_main/axlearn/axlearn/common/flash_attention/neuron_attention_karthick_test.py
# pytest /fsx/mengchiy/fs_main/axlearn/axlearn/common/flash_attention/neuron_attention_test.py

# read_peak_memory

# python -c "import custom_jax_cache" -m /fsx/mengchiy/fs_main/axlearn/test_jax_cache.py