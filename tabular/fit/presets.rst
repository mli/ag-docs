Presets
=======

We you call ``fit``, AutoGluon will explore a set of models to save your
time on manual hyperparameter tuning. The more models to explore, the
better accuracy you often get. But it also leads to more computational
cost. There are several ways to balance the model accuracy and
computational cost. The easiest one is through the ``presets`` argument
in the ``fit`` method.

A preset setting specifies a particular set of models and how they are
combined for prediction. AutoGluon provides 4 presets:
``medium_quality``, ``good_quality``, ``high_quality``, and
``best_quality``. The differences are listed in the following table.

+-----------+-----------+-----------+-----------+-----------+-----------+
| Preset    | Mode      | Fit Time  | Predict   | Disk      | Use Cases |
|           | Quality   |           | Time      | Usage     |           |
+===========+===========+===========+===========+===========+===========+
| ``best_   | Best      | 16x       | 32x       | 16x       | When      |
| quality`` |           |           |           |           | accuracy  |
|           |           |           |           |           | is what   |
|           |           |           |           |           | matters   |
+-----------+-----------+-----------+-----------+-----------+-----------+
| ``high_   | High      | 16x       | 4x        | 2x        | When you  |
| quality`` |           |           |           |           | need a    |
|           |           |           |           |           | very      |
|           |           |           |           |           | powerful  |
|           |           |           |           |           | solution  |
|           |           |           |           |           | with fast |
|           |           |           |           |           | (batch)   |
|           |           |           |           |           | inference |
+-----------+-----------+-----------+-----------+-----------+-----------+
| ``good_   | Good      | 16x       | 2x        | 1x        | When a    |
| quality`` |           |           |           |           | powerful, |
|           |           |           |           |           | highly    |
|           |           |           |           |           | portable  |
|           |           |           |           |           | solution  |
|           |           |           |           |           | with very |
|           |           |           |           |           | fast      |
|           |           |           |           |           | inference |
|           |           |           |           |           | is        |
|           |           |           |           |           | required: |
|           |           |           |           |           | Bill      |
|           |           |           |           |           | ion-scale |
|           |           |           |           |           | batch     |
|           |           |           |           |           | i         |
|           |           |           |           |           | nference, |
|           |           |           |           |           | sub-100ms |
|           |           |           |           |           | online-i  |
|           |           |           |           |           | nference, |
|           |           |           |           |           | edg       |
|           |           |           |           |           | e-devices |
+-----------+-----------+-----------+-----------+-----------+-----------+
| ``medium_ | Medium    | 1x        | 1x        | 1x        | Initial   |
| quality`` |           |           |           |           | pro       |
|           |           |           |           |           | totyping, |
|           |           |           |           |           | est       |
|           |           |           |           |           | ablishing |
|           |           |           |           |           | a         |
|           |           |           |           |           | pe        |
|           |           |           |           |           | rformance |
|           |           |           |           |           | baseline  |
+-----------+-----------+-----------+-----------+-----------+-----------+

We recommend you to start with ``medium_quality``, which is the default
setting, to get a sense of the problem and identify any data related
issues. It’s the fastest option. You can further accelerate it by
subsampling your data or specifying a proper ``time_limit`` argument for
the ``fit`` method.

Once you are comfortable, next try ``best_quality``. Make sure to
specify at least 16x the ``time_limit`` value as used in
``medium_quality``. Once finished, you should have a very powerful
solution that is often stronger than ``medium_quality``, especially for
complex data.

Once you evaluate both ``best_quality`` and ``medium_quality``, check if
either satisfies your needs. If neither do, consider trying
``high_quality`` and/or ``good_quality``.

Now let’s train a model with the ``high_quality`` preset and evaluate
its performance.

.. container:: {toggle}

   .. code:: python

      !pip install autogluon

   ::

      Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
      Collecting autogluon
        Downloading autogluon-0.5.0-py3-none-any.whl (9.5 kB)
      Collecting autogluon.core[all]==0.5.0
        Downloading autogluon.core-0.5.0-py3-none-any.whl (203 kB)
      [K     |████████████████████████████████| 203 kB 3.8 MB/s 
      [?25hCollecting autogluon.vision==0.5.0
        Downloading autogluon.vision-0.5.0-py3-none-any.whl (48 kB)
      [K     |████████████████████████████████| 48 kB 4.2 MB/s 
      [?25hCollecting autogluon.text==0.5.0
        Downloading autogluon.text-0.5.0-py3-none-any.whl (61 kB)
      [K     |████████████████████████████████| 61 kB 246 kB/s 
      [?25hCollecting autogluon.features==0.5.0
        Downloading autogluon.features-0.5.0-py3-none-any.whl (59 kB)
      [K     |████████████████████████████████| 59 kB 1.3 MB/s 
      [?25hCollecting autogluon.multimodal==0.5.0
        Downloading autogluon.multimodal-0.5.0-py3-none-any.whl (141 kB)
      [K     |████████████████████████████████| 141 kB 8.4 MB/s 
      [?25hCollecting autogluon.timeseries[all]==0.5.0
        Downloading autogluon.timeseries-0.5.0-py3-none-any.whl (63 kB)
      [K     |████████████████████████████████| 63 kB 1.8 MB/s 
      [?25hCollecting autogluon.tabular[all]==0.5.0
        Downloading autogluon.tabular-0.5.0-py3-none-any.whl (272 kB)
      [K     |████████████████████████████████| 272 kB 9.3 MB/s 
      [?25hCollecting autogluon.common==0.5.0
        Downloading autogluon.common-0.5.0-py3-none-any.whl (37 kB)
      Requirement already satisfied: scikit-learn<1.1,>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from autogluon.core[all]==0.5.0->autogluon) (1.0.2)
      Collecting distributed<=2021.11.2,>=2021.09.1
        Downloading distributed-2021.11.2-py3-none-any.whl (802 kB)
      [K     |████████████████████████████████| 802 kB 9.1 MB/s 
      [?25hRequirement already satisfied: tqdm>=4.38.0 in /usr/local/lib/python3.7/dist-packages (from autogluon.core[all]==0.5.0->autogluon) (4.64.0)
      Collecting dask<=2021.11.2,>=2021.09.1
        Downloading dask-2021.11.2-py3-none-any.whl (1.0 MB)
      [K     |████████████████████████████████| 1.0 MB 21.7 MB/s 
      [?25hCollecting boto3
        Downloading boto3-1.24.24-py3-none-any.whl (132 kB)
      [K     |████████████████████████████████| 132 kB 32.2 MB/s 
      [?25hRequirement already satisfied: pandas!=1.4.0,<1.5,>=1.2.5 in /usr/local/lib/python3.7/dist-packages (from autogluon.core[all]==0.5.0->autogluon) (1.3.5)
      Requirement already satisfied: numpy<1.23,>=1.21 in /usr/local/lib/python3.7/dist-packages (from autogluon.core[all]==0.5.0->autogluon) (1.21.6)
      Collecting scipy<1.8.0,>=1.5.4
        Downloading scipy-1.7.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (38.1 MB)
      [K     |████████████████████████████████| 38.1 MB 1.2 MB/s 
      [?25hRequirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from autogluon.core[all]==0.5.0->autogluon) (3.2.2)
      Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from autogluon.core[all]==0.5.0->autogluon) (2.23.0)
      Collecting hyperopt<0.2.8,>=0.2.7
        Downloading hyperopt-0.2.7-py2.py3-none-any.whl (1.6 MB)
      [K     |████████████████████████████████| 1.6 MB 27.0 MB/s 
      [?25hCollecting ray<1.14,>=1.13
        Downloading ray-1.13.0-cp37-cp37m-manylinux2014_x86_64.whl (54.5 MB)
      [K     |████████████████████████████████| 54.5 MB 1.4 MB/s 
      [?25hCollecting psutil<6,>=5.7.3
        Downloading psutil-5.9.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (281 kB)
      [K     |████████████████████████████████| 281 kB 33.9 MB/s 
      [?25hCollecting scikit-image<0.20.0,>=0.19.1
        Downloading scikit_image-0.19.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (13.5 MB)
      [K     |████████████████████████████████| 13.5 MB 52.9 MB/s 
      [?25hRequirement already satisfied: torch<1.12,>=1.0 in /usr/local/lib/python3.7/dist-packages (from autogluon.multimodal==0.5.0->autogluon) (1.11.0+cu113)
      Collecting sentencepiece<0.2.0,>=0.1.95
        Downloading sentencepiece-0.1.96-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
      [K     |████████████████████████████████| 1.2 MB 52.2 MB/s 
      [?25hRequirement already satisfied: protobuf<=3.18.1 in /usr/local/lib/python3.7/dist-packages (from autogluon.multimodal==0.5.0->autogluon) (3.17.3)
      Collecting fairscale<0.5.0,>=0.4.5
        Downloading fairscale-0.4.6.tar.gz (248 kB)
      [K     |████████████████████████████████| 248 kB 36.1 MB/s 
      [?25h  Installing build dependencies ... [?25l[?25hdone
        Getting requirements to build wheel ... [?25l[?25hdone
        Installing backend dependencies ... [?25l[?25hdone
          Preparing wheel metadata ... [?25l[?25hdone
      Collecting timm<0.6.0
        Downloading timm-0.5.4-py3-none-any.whl (431 kB)
      [K     |████████████████████████████████| 431 kB 47.6 MB/s 
      [?25hCollecting nlpaug<2.0.0,>=1.1.10
        Downloading nlpaug-1.1.11-py3-none-any.whl (410 kB)
      [K     |████████████████████████████████| 410 kB 47.8 MB/s 
      [?25hCollecting omegaconf<2.2.0,>=2.1.1
        Downloading omegaconf-2.1.2-py3-none-any.whl (74 kB)
      [K     |████████████████████████████████| 74 kB 3.0 MB/s 
      [?25hCollecting Pillow<9.1.0,>=9.0.1
        Downloading Pillow-9.0.1-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.3 MB)
      [K     |████████████████████████████████| 4.3 MB 40.4 MB/s 
      [?25hCollecting nptyping<1.5.0,>=1.4.4
        Downloading nptyping-1.4.4-py3-none-any.whl (31 kB)
      Requirement already satisfied: nltk<4.0.0,>=3.4.5 in /usr/local/lib/python3.7/dist-packages (from autogluon.multimodal==0.5.0->autogluon) (3.7)
      Collecting transformers<4.21.0,>=4.18.0
        Downloading transformers-4.20.1-py3-none-any.whl (4.4 MB)
      [K     |████████████████████████████████| 4.4 MB 45.9 MB/s 
      [?25hCollecting autogluon-contrib-nlp==0.0.1b20220208
        Downloading autogluon_contrib_nlp-0.0.1b20220208-py3-none-any.whl (157 kB)
      [K     |████████████████████████████████| 157 kB 42.9 MB/s 
      [?25hCollecting pytorch-metric-learning<1.4.0,>=1.3.0
        Downloading pytorch_metric_learning-1.3.2-py3-none-any.whl (109 kB)
      [K     |████████████████████████████████| 109 kB 55.7 MB/s 
      [?25hCollecting pytorch-lightning<1.7.0,>=1.5.10
        Downloading pytorch_lightning-1.6.4-py3-none-any.whl (585 kB)
      [K     |████████████████████████████████| 585 kB 32.0 MB/s 
      [?25hCollecting torchmetrics<0.8.0,>=0.7.2
        Downloading torchmetrics-0.7.3-py3-none-any.whl (398 kB)
      [K     |████████████████████████████████| 398 kB 49.5 MB/s 
      [?25hRequirement already satisfied: smart-open<5.3.0,>=5.2.1 in /usr/local/lib/python3.7/dist-packages (from autogluon.multimodal==0.5.0->autogluon) (5.2.1)
      Collecting tokenizers>=0.9.4
        Downloading tokenizers-0.12.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.6 MB)
      [K     |████████████████████████████████| 6.6 MB 50.0 MB/s 
      [?25hCollecting sacrebleu
        Downloading sacrebleu-2.1.0-py3-none-any.whl (92 kB)
      [K     |████████████████████████████████| 92 kB 11.9 MB/s 
      [?25hCollecting yacs>=0.1.6
        Downloading yacs-0.1.8-py3-none-any.whl (14 kB)
      Collecting sacremoses>=0.0.38
        Downloading sacremoses-0.0.53.tar.gz (880 kB)
      [K     |████████████████████████████████| 880 kB 53.3 MB/s 
      [?25hCollecting flake8
        Downloading flake8-4.0.1-py2.py3-none-any.whl (64 kB)
      [K     |████████████████████████████████| 64 kB 2.2 MB/s 
      [?25hCollecting sentencepiece<0.2.0,>=0.1.95
        Downloading sentencepiece-0.1.95-cp37-cp37m-manylinux2014_x86_64.whl (1.2 MB)
      [K     |████████████████████████████████| 1.2 MB 38.4 MB/s 
      [?25hRequirement already satisfied: regex in /usr/local/lib/python3.7/dist-packages (from autogluon-contrib-nlp==0.0.1b20220208->autogluon.multimodal==0.5.0->autogluon) (2022.6.2)
      Requirement already satisfied: pyarrow in /usr/local/lib/python3.7/dist-packages (from autogluon-contrib-nlp==0.0.1b20220208->autogluon.multimodal==0.5.0->autogluon) (6.0.1)
      Collecting contextvars
        Downloading contextvars-2.4.tar.gz (9.6 kB)
      [33mWARNING: autogluon-core 0.5.0 does not provide the extra 'ray-tune'[0m
      Requirement already satisfied: networkx<3.0,>=2.3 in /usr/local/lib/python3.7/dist-packages (from autogluon.tabular[all]==0.5.0->autogluon) (2.6.3)
      Collecting xgboost<1.5,>=1.4
        Downloading xgboost-1.4.2-py3-none-manylinux2010_x86_64.whl (166.7 MB)
      [K     |████████████████████████████████| 166.7 MB 19 kB/s 
      [?25hCollecting catboost<1.1,>=1.0
        Downloading catboost-1.0.6-cp37-none-manylinux1_x86_64.whl (76.6 MB)
      [K     |████████████████████████████████| 76.6 MB 1.3 MB/s 
      [?25hCollecting fastai<2.6,>=2.3.1
        Downloading fastai-2.5.6-py3-none-any.whl (188 kB)
      [K     |████████████████████████████████| 188 kB 61.9 MB/s 
      [?25hCollecting lightgbm<3.4,>=3.3
        Downloading lightgbm-3.3.2-py3-none-manylinux1_x86_64.whl (2.0 MB)
      [K     |████████████████████████████████| 2.0 MB 41.4 MB/s 
      [?25hCollecting gluonts>=0.8.0
        Downloading gluonts-0.10.0-py3-none-any.whl (2.5 MB)
      [K     |████████████████████████████████| 2.5 MB 37.2 MB/s 
      [?25hCollecting psutil<6,>=5.7.3
        Downloading psutil-5.8.0-cp37-cp37m-manylinux2010_x86_64.whl (296 kB)
      [K     |████████████████████████████████| 296 kB 56.7 MB/s 
      [?25hCollecting pmdarima~=1.8
        Downloading pmdarima-1.8.5-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_24_x86_64.whl (1.4 MB)
      [K     |████████████████████████████████| 1.4 MB 32.9 MB/s 
      [?25hCollecting sktime~=0.12
        Downloading sktime-0.12.1-py3-none-any.whl (6.8 MB)
      [K     |████████████████████████████████| 6.8 MB 28.3 MB/s 
      [?25hCollecting tbats~=1.1
        Downloading tbats-1.1.0-py3-none-any.whl (43 kB)
      [K     |████████████████████████████████| 43 kB 2.0 MB/s 
      [?25hCollecting gluoncv<0.10.6,>=0.10.5
        Downloading gluoncv-0.10.5.post0-py2.py3-none-any.whl (1.3 MB)
      [K     |████████████████████████████████| 1.3 MB 38.7 MB/s 
      [?25hRequirement already satisfied: plotly in /usr/local/lib/python3.7/dist-packages (from catboost<1.1,>=1.0->autogluon.tabular[all]==0.5.0->autogluon) (5.5.0)
      Requirement already satisfied: graphviz in /usr/local/lib/python3.7/dist-packages (from catboost<1.1,>=1.0->autogluon.tabular[all]==0.5.0->autogluon) (0.10.1)
      Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from catboost<1.1,>=1.0->autogluon.tabular[all]==0.5.0->autogluon) (1.15.0)
      Collecting partd>=0.3.10
        Downloading partd-1.2.0-py3-none-any.whl (19 kB)
      Collecting fsspec>=0.6.0
        Downloading fsspec-2022.5.0-py3-none-any.whl (140 kB)
      [K     |████████████████████████████████| 140 kB 42.5 MB/s 
      [?25hRequirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from dask<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (21.3)
      Requirement already satisfied: cloudpickle>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from dask<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (1.3.0)
      Requirement already satisfied: toolz>=0.8.2 in /usr/local/lib/python3.7/dist-packages (from dask<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (0.11.2)
      Requirement already satisfied: pyyaml in /usr/local/lib/python3.7/dist-packages (from dask<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (3.13)
      Requirement already satisfied: jinja2 in /usr/local/lib/python3.7/dist-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (2.11.3)
      Requirement already satisfied: zict>=0.1.3 in /usr/local/lib/python3.7/dist-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (2.2.0)
      Requirement already satisfied: sortedcontainers!=2.0.0,!=2.0.1 in /usr/local/lib/python3.7/dist-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (2.4.0)
      Collecting cloudpickle>=1.1.1
        Downloading cloudpickle-2.1.0-py3-none-any.whl (25 kB)
      Requirement already satisfied: click>=6.6 in /usr/local/lib/python3.7/dist-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (7.1.2)
      Requirement already satisfied: msgpack>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (1.0.4)
      Requirement already satisfied: tornado>=5 in /usr/local/lib/python3.7/dist-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (5.1.1)
      Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (57.4.0)
      Requirement already satisfied: tblib>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (1.7.0)
      Requirement already satisfied: fastdownload<2,>=0.0.5 in /usr/local/lib/python3.7/dist-packages (from fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (0.0.6)
      Requirement already satisfied: fastcore<1.5,>=1.3.27 in /usr/local/lib/python3.7/dist-packages (from fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (1.4.4)
      Requirement already satisfied: torchvision>=0.8.2 in /usr/local/lib/python3.7/dist-packages (from fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (0.12.0+cu113)
      Requirement already satisfied: fastprogress>=0.2.4 in /usr/local/lib/python3.7/dist-packages (from fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (1.0.2)
      Requirement already satisfied: spacy<4 in /usr/local/lib/python3.7/dist-packages (from fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (3.3.1)
      Requirement already satisfied: pip in /usr/local/lib/python3.7/dist-packages (from fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (21.1.3)
      Requirement already satisfied: opencv-python in /usr/local/lib/python3.7/dist-packages (from gluoncv<0.10.6,>=0.10.5->autogluon.vision==0.5.0->autogluon) (4.1.2.30)
      Collecting autocfg
        Downloading autocfg-0.0.8-py3-none-any.whl (13 kB)
      Collecting portalocker
        Downloading portalocker-2.4.0-py2.py3-none-any.whl (16 kB)
      Requirement already satisfied: pydantic~=1.7 in /usr/local/lib/python3.7/dist-packages (from gluonts>=0.8.0->autogluon.timeseries[all]==0.5.0->autogluon) (1.8.2)
      Requirement already satisfied: typing-extensions~=4.0 in /usr/local/lib/python3.7/dist-packages (from gluonts>=0.8.0->autogluon.timeseries[all]==0.5.0->autogluon) (4.1.1)
      Requirement already satisfied: holidays>=0.9 in /usr/local/lib/python3.7/dist-packages (from gluonts>=0.8.0->autogluon.timeseries[all]==0.5.0->autogluon) (0.10.5.2)
      Requirement already satisfied: korean-lunar-calendar in /usr/local/lib/python3.7/dist-packages (from holidays>=0.9->gluonts>=0.8.0->autogluon.timeseries[all]==0.5.0->autogluon) (0.2.1)
      Requirement already satisfied: hijri-converter in /usr/local/lib/python3.7/dist-packages (from holidays>=0.9->gluonts>=0.8.0->autogluon.timeseries[all]==0.5.0->autogluon) (2.2.4)
      Requirement already satisfied: python-dateutil in /usr/local/lib/python3.7/dist-packages (from holidays>=0.9->gluonts>=0.8.0->autogluon.timeseries[all]==0.5.0->autogluon) (2.8.2)
      Requirement already satisfied: convertdate>=2.3.0 in /usr/local/lib/python3.7/dist-packages (from holidays>=0.9->gluonts>=0.8.0->autogluon.timeseries[all]==0.5.0->autogluon) (2.4.0)
      Requirement already satisfied: pymeeus<=1,>=0.3.13 in /usr/local/lib/python3.7/dist-packages (from convertdate>=2.3.0->holidays>=0.9->gluonts>=0.8.0->autogluon.timeseries[all]==0.5.0->autogluon) (0.5.11)
      Collecting py4j
        Downloading py4j-0.10.9.5-py2.py3-none-any.whl (199 kB)
      [K     |████████████████████████████████| 199 kB 41.4 MB/s 
      [?25hRequirement already satisfied: future in /usr/local/lib/python3.7/dist-packages (from hyperopt<0.2.8,>=0.2.7->autogluon.core[all]==0.5.0->autogluon) (0.16.0)
      Requirement already satisfied: wheel in /usr/local/lib/python3.7/dist-packages (from lightgbm<3.4,>=3.3->autogluon.tabular[all]==0.5.0->autogluon) (0.37.1)
      Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->autogluon.core[all]==0.5.0->autogluon) (1.4.3)
      Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->autogluon.core[all]==0.5.0->autogluon) (3.0.9)
      Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->autogluon.core[all]==0.5.0->autogluon) (0.11.0)
      Requirement already satisfied: gdown>=4.0.0 in /usr/local/lib/python3.7/dist-packages (from nlpaug<2.0.0,>=1.1.10->autogluon.multimodal==0.5.0->autogluon) (4.4.0)
      Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.7/dist-packages (from gdown>=4.0.0->nlpaug<2.0.0,>=1.1.10->autogluon.multimodal==0.5.0->autogluon) (4.6.3)
      Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from gdown>=4.0.0->nlpaug<2.0.0,>=1.1.10->autogluon.multimodal==0.5.0->autogluon) (3.7.1)
      Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (from nltk<4.0.0,>=3.4.5->autogluon.multimodal==0.5.0->autogluon) (1.1.0)
      Collecting typish>=1.7.0
        Downloading typish-1.9.3-py3-none-any.whl (45 kB)
      [K     |████████████████████████████████| 45 kB 3.0 MB/s 
      [?25hCollecting pyyaml
        Downloading PyYAML-6.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (596 kB)
      [K     |████████████████████████████████| 596 kB 41.5 MB/s 
      [?25hCollecting antlr4-python3-runtime==4.8
        Downloading antlr4-python3-runtime-4.8.tar.gz (112 kB)
      [K     |████████████████████████████████| 112 kB 35.5 MB/s 
      [?25hRequirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas!=1.4.0,<1.5,>=1.2.5->autogluon.core[all]==0.5.0->autogluon) (2022.1)
      Collecting locket
        Downloading locket-1.0.0-py2.py3-none-any.whl (4.4 kB)
      Collecting statsmodels!=0.12.0,>=0.11
        Downloading statsmodels-0.13.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (9.8 MB)
      [K     |████████████████████████████████| 9.8 MB 36.8 MB/s 
      [?25hRequirement already satisfied: urllib3 in /usr/local/lib/python3.7/dist-packages (from pmdarima~=1.8->autogluon.timeseries[all]==0.5.0->autogluon) (1.24.3)
      Requirement already satisfied: Cython!=0.29.18,>=0.29 in /usr/local/lib/python3.7/dist-packages (from pmdarima~=1.8->autogluon.timeseries[all]==0.5.0->autogluon) (0.29.30)
      Collecting pyDeprecate>=0.3.1
        Downloading pyDeprecate-0.3.2-py3-none-any.whl (10 kB)
      Requirement already satisfied: tensorboard>=2.2.0 in /usr/local/lib/python3.7/dist-packages (from pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (2.8.0)
      Collecting aiohttp
        Downloading aiohttp-3.8.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.1 MB)
      [K     |████████████████████████████████| 1.1 MB 37.5 MB/s 
      [?25hRequirement already satisfied: attrs in /usr/local/lib/python3.7/dist-packages (from ray<1.14,>=1.13->autogluon.core[all]==0.5.0->autogluon) (21.4.0)
      Collecting aiosignal
        Downloading aiosignal-1.2.0-py3-none-any.whl (8.2 kB)
      Collecting frozenlist
        Downloading frozenlist-1.3.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (144 kB)
      [K     |████████████████████████████████| 144 kB 54.8 MB/s 
      [?25hCollecting virtualenv
        Downloading virtualenv-20.15.1-py2.py3-none-any.whl (10.1 MB)
      [K     |████████████████████████████████| 10.1 MB 38.8 MB/s 
      [?25hCollecting grpcio<=1.43.0,>=1.28.1
        Downloading grpcio-1.43.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.1 MB)
      [K     |████████████████████████████████| 4.1 MB 36.7 MB/s 
      [?25hRequirement already satisfied: jsonschema in /usr/local/lib/python3.7/dist-packages (from ray<1.14,>=1.13->autogluon.core[all]==0.5.0->autogluon) (4.3.3)
      Requirement already satisfied: tabulate in /usr/local/lib/python3.7/dist-packages (from ray<1.14,>=1.13->autogluon.core[all]==0.5.0->autogluon) (0.8.9)
      Collecting tensorboardX>=1.9
        Downloading tensorboardX-2.5.1-py2.py3-none-any.whl (125 kB)
      [K     |████████████████████████████████| 125 kB 43.2 MB/s 
      [?25hRequirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->autogluon.core[all]==0.5.0->autogluon) (3.0.4)
      Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->autogluon.core[all]==0.5.0->autogluon) (2.10)
      Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->autogluon.core[all]==0.5.0->autogluon) (2022.6.15)
      Requirement already satisfied: PyWavelets>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from scikit-image<0.20.0,>=0.19.1->autogluon.multimodal==0.5.0->autogluon) (1.3.0)
      Requirement already satisfied: tifffile>=2019.7.26 in /usr/local/lib/python3.7/dist-packages (from scikit-image<0.20.0,>=0.19.1->autogluon.multimodal==0.5.0->autogluon) (2021.11.2)
      Requirement already satisfied: imageio>=2.4.1 in /usr/local/lib/python3.7/dist-packages (from scikit-image<0.20.0,>=0.19.1->autogluon.multimodal==0.5.0->autogluon) (2.4.1)
      Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn<1.1,>=1.0.0->autogluon.core[all]==0.5.0->autogluon) (3.1.0)
      Collecting deprecated>=1.2.13
        Downloading Deprecated-1.2.13-py2.py3-none-any.whl (9.6 kB)
      Collecting numba>=0.53
        Downloading numba-0.55.2-cp37-cp37m-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.3 MB)
      [K     |████████████████████████████████| 3.3 MB 45.2 MB/s 
      [?25hRequirement already satisfied: wrapt<2,>=1.10 in /usr/local/lib/python3.7/dist-packages (from deprecated>=1.2.13->sktime~=0.12->autogluon.timeseries[all]==0.5.0->autogluon) (1.14.1)
      Collecting llvmlite<0.39,>=0.38.0rc1
        Downloading llvmlite-0.38.1-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (34.5 MB)
      [K     |████████████████████████████████| 34.5 MB 6.7 kB/s 
      [?25hRequirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (2.0.6)
      Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (3.0.6)
      Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (1.0.7)
      Requirement already satisfied: thinc<8.1.0,>=8.0.14 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (8.0.17)
      Requirement already satisfied: srsly<3.0.0,>=2.4.3 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (2.4.3)
      Requirement already satisfied: langcodes<4.0.0,>=3.2.0 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (3.3.0)
      Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (1.0.2)
      Requirement already satisfied: catalogue<2.1.0,>=2.0.6 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (2.0.7)
      Requirement already satisfied: pathy>=0.3.5 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (0.6.1)
      Requirement already satisfied: blis<0.8.0,>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (0.7.7)
      Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.9 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (3.0.9)
      Requirement already satisfied: wasabi<1.1.0,>=0.9.1 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (0.9.1)
      Requirement already satisfied: typer<0.5.0,>=0.3.0 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (0.4.1)
      Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from catalogue<2.1.0,>=2.0.6->spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (3.8.0)
      Requirement already satisfied: patsy>=0.5.2 in /usr/local/lib/python3.7/dist-packages (from statsmodels!=0.12.0,>=0.11->pmdarima~=1.8->autogluon.timeseries[all]==0.5.0->autogluon) (0.5.2)
      Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (1.35.0)
      Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (1.0.1)
      Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (1.8.1)
      Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (0.4.6)
      Requirement already satisfied: absl-py>=0.4 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (1.1.0)
      Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (0.6.1)
      Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (3.3.7)
      Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (0.2.8)
      Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (4.8)
      Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (4.2.4)
      Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (1.3.1)
      Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (4.11.4)
      Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (0.4.8)
      Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (3.2.0)
      Collecting huggingface-hub<1.0,>=0.1.0
        Downloading huggingface_hub-0.8.1-py3-none-any.whl (101 kB)
      [K     |████████████████████████████████| 101 kB 9.8 MB/s 
      [?25hRequirement already satisfied: heapdict in /usr/local/lib/python3.7/dist-packages (from zict>=0.1.3->distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (1.0.1)
      Collecting yarl<2.0,>=1.0
        Downloading yarl-1.7.2-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (271 kB)
      [K     |████████████████████████████████| 271 kB 37.8 MB/s 
      [?25hCollecting asynctest==0.13.0
        Downloading asynctest-0.13.0-py3-none-any.whl (26 kB)
      Collecting async-timeout<5.0,>=4.0.0a3
        Downloading async_timeout-4.0.2-py3-none-any.whl (5.8 kB)
      Requirement already satisfied: charset-normalizer<3.0,>=2.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->fsspec>=0.6.0->dask<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (2.1.0)
      Collecting multidict<7.0,>=4.5
        Downloading multidict-6.0.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (94 kB)
      [K     |████████████████████████████████| 94 kB 3.3 MB/s 
      [?25hCollecting botocore<1.28.0,>=1.27.24
        Downloading botocore-1.27.24-py3-none-any.whl (9.0 MB)
      [K     |████████████████████████████████| 9.0 MB 37.9 MB/s 
      [?25hCollecting jmespath<2.0.0,>=0.7.1
        Downloading jmespath-1.0.1-py3-none-any.whl (20 kB)
      Collecting s3transfer<0.7.0,>=0.6.0
        Downloading s3transfer-0.6.0-py3-none-any.whl (79 kB)
      [K     |████████████████████████████████| 79 kB 7.1 MB/s 
      [?25hCollecting urllib3
        Downloading urllib3-1.25.11-py2.py3-none-any.whl (127 kB)
      [K     |████████████████████████████████| 127 kB 36.0 MB/s 
      [?25hCollecting immutables>=0.9
        Downloading immutables-0.18-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (116 kB)
      [K     |████████████████████████████████| 116 kB 47.0 MB/s 
      [?25hCollecting pycodestyle<2.9.0,>=2.8.0
        Downloading pycodestyle-2.8.0-py2.py3-none-any.whl (42 kB)
      [K     |████████████████████████████████| 42 kB 865 kB/s 
      [?25hCollecting pyflakes<2.5.0,>=2.4.0
        Downloading pyflakes-2.4.0-py2.py3-none-any.whl (69 kB)
      [K     |████████████████████████████████| 69 kB 7.3 MB/s 
      [?25hCollecting flake8
        Downloading flake8-4.0.0-py2.py3-none-any.whl (64 kB)
      [K     |████████████████████████████████| 64 kB 2.6 MB/s 
      [?25h  Downloading flake8-3.9.2-py2.py3-none-any.whl (73 kB)
      [K     |████████████████████████████████| 73 kB 1.6 MB/s 
      [?25hCollecting pycodestyle<2.8.0,>=2.7.0
        Downloading pycodestyle-2.7.0-py2.py3-none-any.whl (41 kB)
      [K     |████████████████████████████████| 41 kB 542 kB/s 
      [?25hCollecting pyflakes<2.4.0,>=2.3.0
        Downloading pyflakes-2.3.1-py2.py3-none-any.whl (68 kB)
      [K     |████████████████████████████████| 68 kB 6.0 MB/s 
      [?25hCollecting mccabe<0.7.0,>=0.6.0
        Downloading mccabe-0.6.1-py2.py3-none-any.whl (8.6 kB)
      Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.7/dist-packages (from jinja2->distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (2.0.1)
      Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema->ray<1.14,>=1.13->autogluon.core[all]==0.5.0->autogluon) (0.18.1)
      Requirement already satisfied: importlib-resources>=1.4.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema->ray<1.14,>=1.13->autogluon.core[all]==0.5.0->autogluon) (5.7.1)
      Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.7/dist-packages (from plotly->catboost<1.1,>=1.0->autogluon.tabular[all]==0.5.0->autogluon) (8.0.1)
      Requirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /usr/local/lib/python3.7/dist-packages (from requests->autogluon.core[all]==0.5.0->autogluon) (1.7.1)
      Collecting colorama
        Downloading colorama-0.4.5-py2.py3-none-any.whl (16 kB)
      Collecting distlib<1,>=0.3.1
        Downloading distlib-0.3.4-py2.py3-none-any.whl (461 kB)
      [K     |████████████████████████████████| 461 kB 47.9 MB/s 
      [?25hCollecting platformdirs<3,>=2
        Downloading platformdirs-2.5.2-py3-none-any.whl (14 kB)
      Building wheels for collected packages: fairscale, antlr4-python3-runtime, sacremoses, contextvars
        Building wheel for fairscale (PEP 517) ... [?25l[?25hdone
        Created wheel for fairscale: filename=fairscale-0.4.6-py3-none-any.whl size=307252 sha256=d113bb5776308f0bba9982f1e3edbb800356d3a145a102f70ca1555fcf514418
        Stored in directory: /root/.cache/pip/wheels/4e/4f/0b/94c29ea06dfad93260cb0377855f87b7b863312317a7f69fe7
        Building wheel for antlr4-python3-runtime (setup.py) ... [?25l[?25hdone
        Created wheel for antlr4-python3-runtime: filename=antlr4_python3_runtime-4.8-py3-none-any.whl size=141230 sha256=d03ad3557b2da34f825bb7c15e8f1319ba97cc1021d6fb362e0672cf4a422d68
        Stored in directory: /root/.cache/pip/wheels/ca/33/b7/336836125fc9bb4ceaa4376d8abca10ca8bc84ddc824baea6c
        Building wheel for sacremoses (setup.py) ... [?25l[?25hdone
        Created wheel for sacremoses: filename=sacremoses-0.0.53-py3-none-any.whl size=895260 sha256=7d6a59b18d8ebcc16bc8bd62445d75974021beba5f101e411061f05abd8ab0b1
        Stored in directory: /root/.cache/pip/wheels/87/39/dd/a83eeef36d0bf98e7a4d1933a4ad2d660295a40613079bafc9
        Building wheel for contextvars (setup.py) ... [?25l[?25hdone
        Created wheel for contextvars: filename=contextvars-2.4-py3-none-any.whl size=7681 sha256=d0e1404effb8dade9c7e5392bdd66bc4dbb3b8b851e8ad910fe2bf78ed02672c
        Stored in directory: /root/.cache/pip/wheels/0a/11/79/e70e668095c0bb1f94718af672ef2d35ee7a023fee56ef54d9
      Successfully built fairscale antlr4-python3-runtime sacremoses contextvars
      Installing collected packages: urllib3, jmespath, locket, botocore, s3transfer, pyyaml, partd, multidict, fsspec, frozenlist, cloudpickle, yarl, scipy, psutil, dask, boto3, asynctest, async-timeout, aiosignal, pyflakes, pyDeprecate, pycodestyle, portalocker, platformdirs, Pillow, mccabe, immutables, grpcio, distributed, distlib, colorama, autogluon.common, aiohttp, yacs, virtualenv, typish, torchmetrics, tokenizers, statsmodels, sentencepiece, sacremoses, sacrebleu, llvmlite, huggingface-hub, flake8, contextvars, autogluon.core, antlr4-python3-runtime, transformers, timm, tensorboardX, scikit-image, ray, pytorch-metric-learning, pytorch-lightning, py4j, pmdarima, omegaconf, numba, nptyping, nlpaug, gluonts, fairscale, deprecated, autogluon.features, autogluon-contrib-nlp, autocfg, xgboost, tbats, sktime, lightgbm, hyperopt, gluoncv, fastai, catboost, autogluon.timeseries, autogluon.tabular, autogluon.multimodal, autogluon.vision, autogluon.text, autogluon
        Attempting uninstall: urllib3
          Found existing installation: urllib3 1.24.3
          Uninstalling urllib3-1.24.3:
            Successfully uninstalled urllib3-1.24.3
        Attempting uninstall: pyyaml
          Found existing installation: PyYAML 3.13
          Uninstalling PyYAML-3.13:
            Successfully uninstalled PyYAML-3.13
        Attempting uninstall: cloudpickle
          Found existing installation: cloudpickle 1.3.0
          Uninstalling cloudpickle-1.3.0:
            Successfully uninstalled cloudpickle-1.3.0
        Attempting uninstall: scipy
          Found existing installation: scipy 1.4.1
          Uninstalling scipy-1.4.1:
            Successfully uninstalled scipy-1.4.1
        Attempting uninstall: psutil
          Found existing installation: psutil 5.4.8
          Uninstalling psutil-5.4.8:
            Successfully uninstalled psutil-5.4.8
        Attempting uninstall: dask
          Found existing installation: dask 2.12.0
          Uninstalling dask-2.12.0:
            Successfully uninstalled dask-2.12.0
        Attempting uninstall: Pillow
          Found existing installation: Pillow 7.1.2
          Uninstalling Pillow-7.1.2:
            Successfully uninstalled Pillow-7.1.2
        Attempting uninstall: grpcio
          Found existing installation: grpcio 1.46.3
          Uninstalling grpcio-1.46.3:
            Successfully uninstalled grpcio-1.46.3
        Attempting uninstall: distributed
          Found existing installation: distributed 1.25.3
          Uninstalling distributed-1.25.3:
            Successfully uninstalled distributed-1.25.3
        Attempting uninstall: statsmodels
          Found existing installation: statsmodels 0.10.2
          Uninstalling statsmodels-0.10.2:
            Successfully uninstalled statsmodels-0.10.2
        Attempting uninstall: llvmlite
          Found existing installation: llvmlite 0.34.0
          Uninstalling llvmlite-0.34.0:
            Successfully uninstalled llvmlite-0.34.0
        Attempting uninstall: scikit-image
          Found existing installation: scikit-image 0.18.3
          Uninstalling scikit-image-0.18.3:
            Successfully uninstalled scikit-image-0.18.3
        Attempting uninstall: numba
          Found existing installation: numba 0.51.2
          Uninstalling numba-0.51.2:
            Successfully uninstalled numba-0.51.2
        Attempting uninstall: xgboost
          Found existing installation: xgboost 0.90
          Uninstalling xgboost-0.90:
            Successfully uninstalled xgboost-0.90
        Attempting uninstall: lightgbm
          Found existing installation: lightgbm 2.2.3
          Uninstalling lightgbm-2.2.3:
            Successfully uninstalled lightgbm-2.2.3
        Attempting uninstall: hyperopt
          Found existing installation: hyperopt 0.1.2
          Uninstalling hyperopt-0.1.2:
            Successfully uninstalled hyperopt-0.1.2
        Attempting uninstall: fastai
          Found existing installation: fastai 2.6.3
          Uninstalling fastai-2.6.3:
            Successfully uninstalled fastai-2.6.3
      [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
      gym 0.17.3 requires cloudpickle<1.7.0,>=1.2.0, but you have cloudpickle 2.1.0 which is incompatible.
      datascience 0.10.6 requires folium==0.2.1, but you have folium 0.8.3 which is incompatible.
      albumentations 0.1.12 requires imgaug<0.2.7,>=0.2.5, but you have imgaug 0.2.9 which is incompatible.[0m
      Successfully installed Pillow-9.0.1 aiohttp-3.8.1 aiosignal-1.2.0 antlr4-python3-runtime-4.8 async-timeout-4.0.2 asynctest-0.13.0 autocfg-0.0.8 autogluon-0.5.0 autogluon-contrib-nlp-0.0.1b20220208 autogluon.common-0.5.0 autogluon.core-0.5.0 autogluon.features-0.5.0 autogluon.multimodal-0.5.0 autogluon.tabular-0.5.0 autogluon.text-0.5.0 autogluon.timeseries-0.5.0 autogluon.vision-0.5.0 boto3-1.24.24 botocore-1.27.24 catboost-1.0.6 cloudpickle-2.1.0 colorama-0.4.5 contextvars-2.4 dask-2021.11.2 deprecated-1.2.13 distlib-0.3.4 distributed-2021.11.2 fairscale-0.4.6 fastai-2.5.6 flake8-3.9.2 frozenlist-1.3.0 fsspec-2022.5.0 gluoncv-0.10.5.post0 gluonts-0.10.0 grpcio-1.43.0 huggingface-hub-0.8.1 hyperopt-0.2.7 immutables-0.18 jmespath-1.0.1 lightgbm-3.3.2 llvmlite-0.38.1 locket-1.0.0 mccabe-0.6.1 multidict-6.0.2 nlpaug-1.1.11 nptyping-1.4.4 numba-0.55.2 omegaconf-2.1.2 partd-1.2.0 platformdirs-2.5.2 pmdarima-1.8.5 portalocker-2.4.0 psutil-5.8.0 py4j-0.10.9.5 pyDeprecate-0.3.2 pycodestyle-2.7.0 pyflakes-2.3.1 pytorch-lightning-1.6.4 pytorch-metric-learning-1.3.2 pyyaml-6.0 ray-1.13.0 s3transfer-0.6.0 sacrebleu-2.1.0 sacremoses-0.0.53 scikit-image-0.19.3 scipy-1.7.3 sentencepiece-0.1.95 sktime-0.12.1 statsmodels-0.13.2 tbats-1.1.0 tensorboardX-2.5.1 timm-0.5.4 tokenizers-0.12.1 torchmetrics-0.7.3 transformers-4.20.1 typish-1.9.3 urllib3-1.25.11 virtualenv-20.15.1 xgboost-1.4.2 yacs-0.1.8 yarl-1.7.2

.. code:: python

   from autogluon.tabular import TabularDataset, TabularPredictor

   url = 'https://autogluon.s3.amazonaws.com/datasets/Inc/'
   train_data = TabularDataset(url+'train.csv')
   test_data = TabularDataset(url+'test.csv')

   predictor = TabularPredictor(label='class').fit(
       train_data, presets='high_quality')

.. container:: {toggle}

   ::

      No path specified. Models will be saved in: "AutogluonModels/ag-20220707_180104/"
      Presets specified: ['high_quality']
      Beginning AutoGluon training ...
      AutoGluon will save models to "AutogluonModels/ag-20220707_180104/"
      AutoGluon Version:  0.5.0
      Python Version:     3.7.13
      Operating System:   Linux
      Train Data Rows:    39073
      Train Data Columns: 14
      Label Column: class
      Preprocessing data ...
      AutoGluon infers your prediction problem is: 'binary' (because only two unique label-values observed).
          2 unique label values:  [' <=50K', ' >50K']
          If 'binary' is not the correct problem_type, please manually specify the problem_type parameter during predictor init (You may specify problem_type as one of: ['binary', 'multiclass', 'regression'])
      Selected class <--> label mapping:  class 1 =  >50K, class 0 =  <=50K
          Note: For your binary classification, AutoGluon arbitrarily selected which label-value represents positive ( >50K) vs negative ( <=50K) class.
          To explicitly set the positive_class, either rename classes to 1 and 0, or specify positive_class in Predictor init.
      Using Feature Generators to preprocess the data ...
      Fitting AutoMLPipelineFeatureGenerator...
          Available Memory:                    11967.12 MB
          Train Data (Original)  Memory Usage: 22.92 MB (0.2% of available memory)
          Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
          Stage 1 Generators:
              Fitting AsTypeFeatureGenerator...
                  Note: Converting 1 features to boolean dtype as they only contain 2 unique values.
          Stage 2 Generators:
              Fitting FillNaFeatureGenerator...
          Stage 3 Generators:
              Fitting IdentityFeatureGenerator...
              Fitting CategoryFeatureGenerator...
                  Fitting CategoryMemoryMinimizeFeatureGenerator...
          Stage 4 Generators:
              Fitting DropUniqueFeatureGenerator...
          Types of features in original data (raw dtype, special dtypes):
              ('int', [])    : 6 | ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', ...]
              ('object', []) : 8 | ['workclass', 'education', 'marital-status', 'occupation', 'relationship', ...]
          Types of features in processed data (raw dtype, special dtypes):
              ('category', [])  : 7 | ['workclass', 'education', 'marital-status', 'occupation', 'relationship', ...]
              ('int', [])       : 6 | ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', ...]
              ('int', ['bool']) : 1 | ['sex']
          0.4s = Fit runtime
          14 features in original data used to generate 14 features in processed data.
          Train Data (Processed) Memory Usage: 2.19 MB (0.0% of available memory)
      Data preprocessing and feature engineering runtime = 0.6s ...
      AutoGluon will gauge predictive performance using evaluation metric: 'accuracy'
          To change this, specify the eval_metric parameter of Predictor()
      AutoGluon will fit 2 stack levels (L1 to L2) ...
      Fitting 13 L1 models ...
      Fitting model: KNeighborsUnif_BAG_L1 ...
          0.7775   = Validation score   (accuracy)
          0.06s    = Training   runtime
          0.41s    = Validation runtime
      Fitting model: KNeighborsDist_BAG_L1 ...
          0.7728   = Validation score   (accuracy)
          0.05s    = Training   runtime
          0.41s    = Validation runtime
      Fitting model: LightGBMXT_BAG_L1 ...
          Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
          0.8683   = Validation score   (accuracy)
          26.35s   = Training   runtime
          1.54s    = Validation runtime
      Fitting model: LightGBM_BAG_L1 ...
          Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
          0.8745   = Validation score   (accuracy)
          23.46s   = Training   runtime
          0.81s    = Validation runtime
      Fitting model: RandomForestGini_BAG_L1 ...
          0.8564   = Validation score   (accuracy)
          12.41s   = Training   runtime
          2.14s    = Validation runtime
      Fitting model: RandomForestEntr_BAG_L1 ...
          0.8581   = Validation score   (accuracy)
          14.5s    = Training   runtime
          2.0s     = Validation runtime
      Fitting model: CatBoost_BAG_L1 ...
          Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
          0.8738   = Validation score   (accuracy)
          273.06s  = Training   runtime
          0.43s    = Validation runtime
      Fitting model: ExtraTreesGini_BAG_L1 ...
          0.8507   = Validation score   (accuracy)
          7.69s    = Training   runtime
          2.17s    = Validation runtime
      Fitting model: ExtraTreesEntr_BAG_L1 ...
          0.8507   = Validation score   (accuracy)
          8.29s    = Training   runtime
          2.17s    = Validation runtime
      Fitting model: NeuralNetFastAI_BAG_L1 ...
          Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
          0.86     = Validation score   (accuracy)
          385.07s  = Training   runtime
          1.39s    = Validation runtime
      Fitting model: XGBoost_BAG_L1 ...
          Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
          0.8753   = Validation score   (accuracy)
          24.16s   = Training   runtime
          0.58s    = Validation runtime
      Fitting model: NeuralNetTorch_BAG_L1 ...
          Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
          0.8578   = Validation score   (accuracy)
          551.93s  = Training   runtime
          1.69s    = Validation runtime
      Fitting model: LightGBMLarge_BAG_L1 ...
          Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
          0.8737   = Validation score   (accuracy)
          31.65s   = Training   runtime
          1.75s    = Validation runtime
      Fitting model: WeightedEnsemble_L2 ...
          0.8753   = Validation score   (accuracy)
          18.4s    = Training   runtime
          0.08s    = Validation runtime
      Fitting 11 L2 models ...
      Fitting model: LightGBMXT_BAG_L2 ...
          Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
          0.8762   = Validation score   (accuracy)
          25.01s   = Training   runtime
          0.41s    = Validation runtime
      Fitting model: LightGBM_BAG_L2 ...
          Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
          0.8763   = Validation score   (accuracy)
          24.43s   = Training   runtime
          0.29s    = Validation runtime
      Fitting model: RandomForestGini_BAG_L2 ...
          0.8759   = Validation score   (accuracy)
          28.76s   = Training   runtime
          1.94s    = Validation runtime
      Fitting model: RandomForestEntr_BAG_L2 ...
          0.8752   = Validation score   (accuracy)
          43.22s   = Training   runtime
          1.94s    = Validation runtime
      Fitting model: CatBoost_BAG_L2 ...
          Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
          0.8761   = Validation score   (accuracy)
          94.17s   = Training   runtime
          0.31s    = Validation runtime
      Fitting model: ExtraTreesGini_BAG_L2 ...
          0.8748   = Validation score   (accuracy)
          9.13s    = Training   runtime
          2.17s    = Validation runtime
      Fitting model: ExtraTreesEntr_BAG_L2 ...
          0.8755   = Validation score   (accuracy)
          9.93s    = Training   runtime
          2.17s    = Validation runtime
      Fitting model: NeuralNetFastAI_BAG_L2 ...
          Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
          0.876    = Validation score   (accuracy)
          357.21s  = Training   runtime
          1.5s     = Validation runtime
      Fitting model: XGBoost_BAG_L2 ...
          Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
          0.8766   = Validation score   (accuracy)
          26.02s   = Training   runtime
          0.47s    = Validation runtime
      Fitting model: NeuralNetTorch_BAG_L2 ...
          Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
          0.8752   = Validation score   (accuracy)
          413.67s  = Training   runtime
          1.85s    = Validation runtime
      Fitting model: LightGBMLarge_BAG_L2 ...
          Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
          0.8764   = Validation score   (accuracy)
          35.02s   = Training   runtime
          0.82s    = Validation runtime
      Fitting model: WeightedEnsemble_L3 ...
          0.8767   = Validation score   (accuracy)
          15.56s   = Training   runtime
          0.07s    = Validation runtime
      AutoGluon training complete, total runtime = 2534.02s ... Best model: "WeightedEnsemble_L3"
      Fitting model: KNeighborsUnif_BAG_L1_FULL | Skipping fit via cloning parent ...
          0.06s    = Training   runtime
          0.41s    = Validation runtime
      Fitting model: KNeighborsDist_BAG_L1_FULL | Skipping fit via cloning parent ...
          0.05s    = Training   runtime
          0.41s    = Validation runtime
      Fitting 1 L1 models ...
      Fitting model: LightGBMXT_BAG_L1_FULL ...
          1.31s    = Training   runtime
      Fitting 1 L1 models ...
      Fitting model: LightGBM_BAG_L1_FULL ...
          1.08s    = Training   runtime
      Fitting model: RandomForestGini_BAG_L1_FULL | Skipping fit via cloning parent ...
          12.41s   = Training   runtime
          2.14s    = Validation runtime
      Fitting model: RandomForestEntr_BAG_L1_FULL | Skipping fit via cloning parent ...
          14.5s    = Training   runtime
          2.0s     = Validation runtime
      Fitting 1 L1 models ...
      Fitting model: CatBoost_BAG_L1_FULL ...
          31.05s   = Training   runtime
      Fitting model: ExtraTreesGini_BAG_L1_FULL | Skipping fit via cloning parent ...
          7.69s    = Training   runtime
          2.17s    = Validation runtime
      Fitting model: ExtraTreesEntr_BAG_L1_FULL | Skipping fit via cloning parent ...
          8.29s    = Training   runtime
          2.17s    = Validation runtime
      Fitting 1 L1 models ...
      Fitting model: NeuralNetFastAI_BAG_L1_FULL ...
          Stopping at the best epoch learned earlier - 15.
          30.3s    = Training   runtime
      Fitting 1 L1 models ...
      Fitting model: XGBoost_BAG_L1_FULL ...
          1.42s    = Training   runtime
      Fitting 1 L1 models ...
      Fitting model: NeuralNetTorch_BAG_L1_FULL ...
          41.11s   = Training   runtime
      Fitting 1 L1 models ...
      Fitting model: LightGBMLarge_BAG_L1_FULL ...
          1.64s    = Training   runtime
      Fitting model: WeightedEnsemble_L2_FULL | Skipping fit via cloning parent ...
          18.4s    = Training   runtime
      Fitting 1 L2 models ...
      Fitting model: LightGBMXT_BAG_L2_FULL ...
          0.93s    = Training   runtime
      Fitting 1 L2 models ...
      Fitting model: LightGBM_BAG_L2_FULL ...
          0.89s    = Training   runtime
      Fitting model: RandomForestGini_BAG_L2_FULL | Skipping fit via cloning parent ...
          28.76s   = Training   runtime
          1.94s    = Validation runtime
      Fitting model: RandomForestEntr_BAG_L2_FULL | Skipping fit via cloning parent ...
          43.22s   = Training   runtime
          1.94s    = Validation runtime
      Fitting 1 L2 models ...
      Fitting model: CatBoost_BAG_L2_FULL ...
          2.7s     = Training   runtime
      Fitting model: ExtraTreesGini_BAG_L2_FULL | Skipping fit via cloning parent ...
          9.13s    = Training   runtime
          2.17s    = Validation runtime
      Fitting model: ExtraTreesEntr_BAG_L2_FULL | Skipping fit via cloning parent ...
          9.93s    = Training   runtime
          2.17s    = Validation runtime
      Fitting 1 L2 models ...
      Fitting model: NeuralNetFastAI_BAG_L2_FULL ...
          Stopping at the best epoch learned earlier - 9.
          19.45s   = Training   runtime
      Fitting 1 L2 models ...
      Fitting model: XGBoost_BAG_L2_FULL ...
          1.4s     = Training   runtime
      Fitting 1 L2 models ...
      Fitting model: NeuralNetTorch_BAG_L2_FULL ...
          23.64s   = Training   runtime
      Fitting 1 L2 models ...
      Fitting model: LightGBMLarge_BAG_L2_FULL ...
          1.94s    = Training   runtime
      Fitting model: WeightedEnsemble_L3_FULL | Skipping fit via cloning parent ...
          15.56s   = Training   runtime
      TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220707_180104/")

.. code:: python

    predictor.evaluate(test_data, silent=True)




.. parsed-literal::

    {'accuracy': 0.8774695465247211,
     'balanced_accuracy': 0.7975275683791567,
     'f1': 0.7142516113630938,
     'mcc': 0.643318065933343,
     'precision': 0.7995724211651524,
     'recall': 0.6453839516824849,
     'roc_auc': 0.9327997272719587}



You can see the accuracy is slighted increased compared to the default
``medium_quality`` preset in :doc:`/get_started/tabular_quick_start`.
But note that we are using a very simple dataset, this small difference
is expected.

Finally, if none of the presets satisfy your requirements, can you
manually specify the set of models to fit with their hyperparameters.
Refer to TODO.
