# Functional Flow Matching

This repo contains the code for our paper [Functional Flow Matching](https://arxiv.org/abs/2305.17209) (Oral, AISTATS 2024).

Our code is roughly structured as follows:
- `/configs` contains example configuration files that can be used to configure parameters for experiments
- `/data` contains the data used in our experiments
- `/models` contains the various neural architectures used in this work
- `/scripts` has files that can be used to launch training jobs to reproduce our experiments
- `/util` contains various utilities, e.g. for reading config files and performing evaluation
- The files `functional_fm.py` and `conditional_ffm.py` implement our FFM model.
- The file `diffusion.py` and `losses.py` implement the baseline DDPM and DDO models. Similarly, `gano.py` and `gano1d.py` implement the GANO baseline.

# Citation

If you found our code useful or build upon our work, we ask that you cite our AISTATS 2024 paper as follows:

```
@inproceedings{kerrigan2024functional,
  title = {Functional FLow Matching,
  author = {Gavin Kerrigan and Giosue Migliorini and Padhraic Smyth},
  booktitle = {The 27th International Conference on AI and Statistics (AISTATS)},
  year = {2024}
}
```
