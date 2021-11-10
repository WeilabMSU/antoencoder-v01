## Requirments

OS Requirements
- CentOS Linux 7 (Core)

Python Dependencies
- setuptools (>=18.0)
- python (>=3.7)
- pytorch (>=1.2)
- rdkit (2020.03)
- biopandas (0.2.7)
- numpy (1.17.4)
- scikit-learn (0.23.2)
- scipy (1.5.2)
- pandas (0.25.3)

## A LV-FP generation example:
python3 generate_bt_fps-lstm.py --model_name_or_path data-bin/smiles --checkpoint_file checkpoint_last.pt --data_name_or_path  data-bin/smiles --target_file example-smiles/example.smi --save_feature_path example-smiles/examples_lv_fp.npy
