#!/bin/bash
# Sequential adaptive-vs-uniform comparison: 8 threads on the 8 physical P-cores, one plan at a time,
# stopping if the job-scheduler starts a benchmark job. Results: sweeps/cmp_<name>.csv/.log
source /opt/software/miniforge/latest/etc/profile.d/conda.sh
conda activate magtense-cu13
set -u
unset OMP_PLACES
export OMP_PROC_BIND=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
export CDFMM_CACHE_DIR="$PWD/.cache"
CPUS=0,2,4,6,8,10,12,14
M=/home/mihaa/MagTense/MagTense_private/python/experiments/grain_dipfmm/output
run() {
    name=$1; shift
    echo "=== $name $(date +%H:%M:%S)"
    PYTHONPATH=build taskset -c $CPUS python -u examples/validation/adaptive_voronoi_sweep.py --stop-if-busy --repeats 5 \
        --csv sweeps/cmp_$name.csv "$@" > sweeps/cmp_$name.log 2>&1
    rc=$?
    grep -v "^\[cdfmm\] Uniform\|^  \|^\[cdfmm\] warning" sweeps/cmp_$name.log | tail -40
    echo "=== $name exit $rc $(date +%H:%M:%S)"
    [ $rc -eq 3 ] && exit 3
}
run smoke --npz $M/smoke_test/smoke_adaptive_mesh.npz --orders 4 6 8 --capacities 16 32 64 --depths 5 --uniform-depths 2 3 4 --targets 1024
run magtense_lev1 --npz $M/adaptive/grain_dipfmm_L400nm_n60x60x60_g200_seed4_w8.3333nm_cone5deg_base16_lev1.npz \
    --orders 4 6 8 --capacities 16 32 64 --depths 5 --uniform-depths 3 4 5 --targets 1024
run synthetic_b16_l2 --grains 30 --base 16 --levels 2 --band 0.02 --orders 4 6 8 --capacities 16 32 64 --depths 5 6 \
    --uniform-depths 3 4 5 --targets 1024
run synthetic_b12_l1 --grains 12 --base 12 --levels 1 --band 0.04 --orders 6 --capacities 32 --depths 5 --uniform-depths 2 3 --targets 512
run magtense_lev2_fp32 --npz $M/adaptive/grain_dipfmm_L400nm_n60x60x60_g200_seed4_w8.3333nm_cone5deg_base16_lev2.npz \
    --fp32 --orders 6 --capacities 16 32 64 --depths 5 6 --uniform-depths 4 5 --targets 1024
echo "=== all done $(date +%H:%M:%S)"
