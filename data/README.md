# Data sources

Raw data is not bundled in this repository. Download it from the original
sources below and place it in this `data/` directory in the layout shown.

## 1. Alibaba 2023 GPU Cluster Trace (Lingjun release)

Source: https://github.com/alibaba/alibaba-lingjun-dataset-2023

Download `job.csv`, `worker.csv`, and `topo.csv` from that repository and
place them directly under `data/`:

```
data/job.csv
data/worker.csv
data/topo.csv
```

License: see the source repository's own license terms at the URL above.
This is the primary trace the paper's headline results (Sections 6.1-6.3)
are built from.

## 2. Helios GPU Cluster Trace (cross-trace validation, Section 6.4 / 7)

Source: https://github.com/S-Lab-System-Group/HeliosData
License: CC-BY-4.0 (stated in that repository's `LICENSE.txt`)
Citation: this is the trace used in Hu et al., "Characterization and
Prediction of Deep Learning Workloads in Large-Scale GPU Datacenters,"
Proc. SC, 2021 (paper reference [6]).

Clone the repository, unzip `data.zip`, and place the four cluster folders
under `data/helios/`:

```
data/helios/Earth/cluster_log.csv
data/helios/Earth/cluster_gpu_number.csv
data/helios/Saturn/cluster_log.csv
data/helios/Saturn/cluster_gpu_number.csv
data/helios/Uranus/cluster_log.csv
data/helios/Uranus/cluster_gpu_number.csv
data/helios/Venus/cluster_log.csv
data/helios/Venus/cluster_gpu_number.csv
```

```bash
git clone https://github.com/S-Lab-System-Group/HeliosData.git
unzip HeliosData/data.zip -d data/helios_tmp
mv data/helios_tmp/data/* data/helios/
rm -rf data/helios_tmp HeliosData
```

## Verifying you have the right layout

After both downloads, `scripts/01_build_dataset.py` should run without a
`FileNotFoundError`, and `scripts/07_crosstrace_helios.py` should print
real job counts for all four Helios clusters when it starts.
