To run single layer str for fixed params:
sbatch final_str.sh




grid_search_str.py: Reads in LH learning rates and runs the array num.
grid_search_str_copy.py: :, Also for dense

(train)_model_smep_str.py: Model called.
external_world.py: 28x28, 20x20 Reduced

(train)_model_smep_str_dense.py: Model called.

SBATCH Files:
gs4_sdhybrid


Results Locations:
smep_lay1_r4: Results from when train_model_str was directly called.
GS_Run50ep: LH Grid search results
scratchlfs02/
-...nps20: 
-...nps4: Full28x28, Reduced20x20
