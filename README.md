An implementation of baselines for multimodal utterance embedding, as described in <https://www.aclweb.org/anthology/N19-1267>.

Based on [original SIF implementation)](https://github.com/PrincetonML/SIF) by Arora et al. (2016, 2017).

Requires Python 3.

# Data

Processed data for the MOSI and POM datasets used in the code can be obtained from [here](https://drive.google.com/drive/folders/1JhCxsNgYB1brG6-e7mNJhMR8fmOCtq_6?usp=sharing), and should be saved in the folder `data/`. Alternatively, you can get the raw data [here](https://github.com/A2Zadeh/CMU-MultimodalSDK).

# Instructions

`configs/` contains JSON files that holds the hyperparameters of the model. To generate some config files, run `python configs/make_configs.py`. These will be saved in `configs/multimodal_search`.

Then, to run MMB2,
```
python simplesif.py configs/multimodal_search/config_0.json $DATASET
```
where `$DATASET` is `mosi` or `pom`.

For MMB1, set the `--unimodal` flag:
```
python simplesif.py configs/multimodal_search/config_0.json $DATASET --unimodal
```

Run `python simplesif.py --help` for more options`.

# License

This code is released under the MIT License. See LICENSE for more details.
