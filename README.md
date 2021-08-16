# STF 

This is a PyTorch and TensorLy implementation of **Accurate Online Tensor Factorization for Temporal Tensor Streams with Missing Values** (CIKM 2021).<br>


## Prerequisites

- Python 3.6+
- [DotMap](https://pypi.org/project/dotmap/)
- [NumPy](https://numpy.org)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [PyTorch](https://pytorch.org/)
- [TensorLy](http://tensorly.org/stable/index.html)


## Usage

- Install all of the prerequisites
- You can run the demo script by `bash demo.sh`, which simply runs `src/main.py`.
- You can change the dataset by modifying `src/main.py` and check the dataset in `data` directory.
- You can change the number of hyper-parameters by modifying `src/stf.py`.
- you can check out the running results in `out` directory, and then plot the results.


## Datasets
- There are six data files stored in COO format (e.g. i j k value), and data statistics as follows.

|         Name        |          Description          |      Size      |   NNZ  | Granularity in Time |                                    Original Source                                   |
|:-------------------:|:-----------------------------:|:--------------:|:------:|:-------------------:|:------------------------------------------------------------------------------------:|
| Beijing Air Quality | locations x pollutants x time | 12 x 6 x 5994  | 618835 | hourly              | [Link](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Datal) |
| Madrid Air Quality  | locations x pollutants x time | 26 x 17 x 3043 | 383279 | hourly              | [Link](https://www.kaggle.com/decide-soluciones/air-quality-madrid)                  |
| Radar Traffic       | locations x directions x time | 17 x 5 x 6419  | 181719 | hourly              | [Link](https://www.kaggle.com/vinayshanbhag/radar-traffic-data)                      |
| Indoor Condition    | locations x sensor x time     | 9 x 2 x 2622   | 59220  | hourly              | [Link](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)         |
| Intel Lab Sensor    | locations x sensor x time     | 54 x 4 x 1152  | 513508 | every 10 minutes    | [Link](http://db.csail.mit.edu/labdata/labdata.html)                                 |
| Chicago Taxi        | sources x destinations x time | 77 x 77 x 2904 | 424440 | hourly              | [Link](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)           |
