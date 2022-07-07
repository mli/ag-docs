Tabluar Prediction
==================

In a tabular prediction task, we predict the values in a column based on
the rest columns’ values. This tutorial demonstrates how to use
AutoGluon for this task.

To start, import the :class:`autogluon.tabular.TabularDataset` and
:class:`autogluon.tabular.TabularPredictor` classes. We will use the
former to load data and the latter to train models and predict.

.. container:: {toggle}

   .. code:: python

      !pip install autogluon

   ::

      Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
      Collecting autogluon
        Downloading autogluon-0.5.0-py3-none-any.whl (9.5 kB)
      Collecting autogluon.features==0.5.0
        Downloading autogluon.features-0.5.0-py3-none-any.whl (59 kB)
      [K     |████████████████████████████████| 59 kB 7.3 MB/s 
      [?25hCollecting autogluon.timeseries[all]==0.5.0
        Downloading autogluon.timeseries-0.5.0-py3-none-any.whl (63 kB)
      [K     |████████████████████████████████| 63 kB 2.6 MB/s 
      [?25hCollecting autogluon.core[all]==0.5.0
        Downloading autogluon.core-0.5.0-py3-none-any.whl (203 kB)
      [K     |████████████████████████████████| 203 kB 69.6 MB/s 
      [?25hCollecting autogluon.text==0.5.0
        Downloading autogluon.text-0.5.0-py3-none-any.whl (61 kB)
      [K     |████████████████████████████████| 61 kB 299 kB/s 
      [?25hCollecting autogluon.vision==0.5.0
        Downloading autogluon.vision-0.5.0-py3-none-any.whl (48 kB)
      [K     |████████████████████████████████| 48 kB 6.0 MB/s 
      [?25hCollecting autogluon.tabular[all]==0.5.0
        Downloading autogluon.tabular-0.5.0-py3-none-any.whl (272 kB)
      [K     |████████████████████████████████| 272 kB 57.9 MB/s 
      [?25hCollecting autogluon.multimodal==0.5.0
        Downloading autogluon.multimodal-0.5.0-py3-none-any.whl (141 kB)
      [K     |████████████████████████████████| 141 kB 66.0 MB/s 
      [?25hCollecting distributed<=2021.11.2,>=2021.09.1
        Downloading distributed-2021.11.2-py3-none-any.whl (802 kB)
      [K     |████████████████████████████████| 802 kB 61.3 MB/s 
      [?25hCollecting autogluon.common==0.5.0
        Downloading autogluon.common-0.5.0-py3-none-any.whl (37 kB)
      Collecting boto3
        Downloading boto3-1.24.24-py3-none-any.whl (132 kB)
      [K     |████████████████████████████████| 132 kB 60.7 MB/s 
      [?25hRequirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from autogluon.core[all]==0.5.0->autogluon) (2.23.0)
      Requirement already satisfied: scikit-learn<1.1,>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from autogluon.core[all]==0.5.0->autogluon) (1.0.2)
      Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from autogluon.core[all]==0.5.0->autogluon) (3.2.2)
      Collecting dask<=2021.11.2,>=2021.09.1
        Downloading dask-2021.11.2-py3-none-any.whl (1.0 MB)
      [K     |████████████████████████████████| 1.0 MB 59.7 MB/s 
      [?25hRequirement already satisfied: pandas!=1.4.0,<1.5,>=1.2.5 in /usr/local/lib/python3.7/dist-packages (from autogluon.core[all]==0.5.0->autogluon) (1.3.5)
      Requirement already satisfied: tqdm>=4.38.0 in /usr/local/lib/python3.7/dist-packages (from autogluon.core[all]==0.5.0->autogluon) (4.64.0)
      Requirement already satisfied: numpy<1.23,>=1.21 in /usr/local/lib/python3.7/dist-packages (from autogluon.core[all]==0.5.0->autogluon) (1.21.6)
      Collecting scipy<1.8.0,>=1.5.4
        Downloading scipy-1.7.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (38.1 MB)
      [K     |████████████████████████████████| 38.1 MB 1.3 MB/s 
      [?25hCollecting hyperopt<0.2.8,>=0.2.7
        Downloading hyperopt-0.2.7-py2.py3-none-any.whl (1.6 MB)
      [K     |████████████████████████████████| 1.6 MB 44.7 MB/s 
      [?25hCollecting ray<1.14,>=1.13
        Downloading ray-1.13.0-cp37-cp37m-manylinux2014_x86_64.whl (54.5 MB)
      [K     |████████████████████████████████| 54.5 MB 226 kB/s 
      [?25hCollecting psutil<6,>=5.7.3
        Downloading psutil-5.9.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (281 kB)
      [K     |████████████████████████████████| 281 kB 73.2 MB/s 
      [?25hCollecting transformers<4.21.0,>=4.18.0
        Downloading transformers-4.20.1-py3-none-any.whl (4.4 MB)
      [K     |████████████████████████████████| 4.4 MB 50.7 MB/s 
      [?25hRequirement already satisfied: nltk<4.0.0,>=3.4.5 in /usr/local/lib/python3.7/dist-packages (from autogluon.multimodal==0.5.0->autogluon) (3.7)
      Collecting scikit-image<0.20.0,>=0.19.1
        Downloading scikit_image-0.19.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (13.5 MB)
      [K     |████████████████████████████████| 13.5 MB 31.1 MB/s 
      [?25hCollecting pytorch-metric-learning<1.4.0,>=1.3.0
        Downloading pytorch_metric_learning-1.3.2-py3-none-any.whl (109 kB)
      [K     |████████████████████████████████| 109 kB 65.1 MB/s 
      [?25hCollecting fairscale<0.5.0,>=0.4.5
        Downloading fairscale-0.4.6.tar.gz (248 kB)
      [K     |████████████████████████████████| 248 kB 69.8 MB/s 
      [?25h  Installing build dependencies ... [?25l[?25hdone
        Getting requirements to build wheel ... [?25l[?25hdone
        Installing backend dependencies ... [?25l[?25hdone
          Preparing wheel metadata ... [?25l[?25hdone
      Requirement already satisfied: protobuf<=3.18.1 in /usr/local/lib/python3.7/dist-packages (from autogluon.multimodal==0.5.0->autogluon) (3.17.3)
      Collecting omegaconf<2.2.0,>=2.1.1
        Downloading omegaconf-2.1.2-py3-none-any.whl (74 kB)
      [K     |████████████████████████████████| 74 kB 4.2 MB/s 
      [?25hCollecting Pillow<9.1.0,>=9.0.1
        Downloading Pillow-9.0.1-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.3 MB)
      [K     |████████████████████████████████| 4.3 MB 44.6 MB/s 
      [?25hRequirement already satisfied: torch<1.12,>=1.0 in /usr/local/lib/python3.7/dist-packages (from autogluon.multimodal==0.5.0->autogluon) (1.11.0+cu113)
      Requirement already satisfied: smart-open<5.3.0,>=5.2.1 in /usr/local/lib/python3.7/dist-packages (from autogluon.multimodal==0.5.0->autogluon) (5.2.1)
      Collecting nlpaug<2.0.0,>=1.1.10
        Downloading nlpaug-1.1.11-py3-none-any.whl (410 kB)
      [K     |████████████████████████████████| 410 kB 70.6 MB/s 
      [?25hCollecting sentencepiece<0.2.0,>=0.1.95
        Downloading sentencepiece-0.1.96-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
      [K     |████████████████████████████████| 1.2 MB 63.4 MB/s 
      [?25hCollecting torchmetrics<0.8.0,>=0.7.2
        Downloading torchmetrics-0.7.3-py3-none-any.whl (398 kB)
      [K     |████████████████████████████████| 398 kB 52.5 MB/s 
      [?25hCollecting nptyping<1.5.0,>=1.4.4
        Downloading nptyping-1.4.4-py3-none-any.whl (31 kB)
      Collecting pytorch-lightning<1.7.0,>=1.5.10
        Downloading pytorch_lightning-1.6.4-py3-none-any.whl (585 kB)
      [K     |████████████████████████████████| 585 kB 46.7 MB/s 
      [?25hCollecting timm<0.6.0
        Downloading timm-0.5.4-py3-none-any.whl (431 kB)
      [K     |████████████████████████████████| 431 kB 49.9 MB/s 
      [?25hCollecting autogluon-contrib-nlp==0.0.1b20220208
        Downloading autogluon_contrib_nlp-0.0.1b20220208-py3-none-any.whl (157 kB)
      [K     |████████████████████████████████| 157 kB 50.5 MB/s 
      [?25hCollecting sentencepiece<0.2.0,>=0.1.95
        Downloading sentencepiece-0.1.95-cp37-cp37m-manylinux2014_x86_64.whl (1.2 MB)
      [K     |████████████████████████████████| 1.2 MB 56.8 MB/s 
      [?25hCollecting yacs>=0.1.6
        Downloading yacs-0.1.8-py3-none-any.whl (14 kB)
      Collecting sacremoses>=0.0.38
        Downloading sacremoses-0.0.53.tar.gz (880 kB)
      [K     |████████████████████████████████| 880 kB 50.1 MB/s 
      [?25hCollecting sacrebleu
        Downloading sacrebleu-2.1.0-py3-none-any.whl (92 kB)
      [K     |████████████████████████████████| 92 kB 11.4 MB/s 
      [?25hCollecting tokenizers>=0.9.4
        Downloading tokenizers-0.12.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.6 MB)
      [K     |████████████████████████████████| 6.6 MB 48.5 MB/s 
      [?25hRequirement already satisfied: pyarrow in /usr/local/lib/python3.7/dist-packages (from autogluon-contrib-nlp==0.0.1b20220208->autogluon.multimodal==0.5.0->autogluon) (6.0.1)
      Collecting contextvars
        Downloading contextvars-2.4.tar.gz (9.6 kB)
      Collecting flake8
        Downloading flake8-4.0.1-py2.py3-none-any.whl (64 kB)
      [K     |████████████████████████████████| 64 kB 3.0 MB/s 
      [?25hRequirement already satisfied: regex in /usr/local/lib/python3.7/dist-packages (from autogluon-contrib-nlp==0.0.1b20220208->autogluon.multimodal==0.5.0->autogluon) (2022.6.2)
      [33mWARNING: autogluon-core 0.5.0 does not provide the extra 'ray-tune'[0m
      Requirement already satisfied: networkx<3.0,>=2.3 in /usr/local/lib/python3.7/dist-packages (from autogluon.tabular[all]==0.5.0->autogluon) (2.6.3)
      Collecting catboost<1.1,>=1.0
        Downloading catboost-1.0.6-cp37-none-manylinux1_x86_64.whl (76.6 MB)
      [K     |████████████████████████████████| 76.6 MB 1.2 MB/s 
      [?25hCollecting fastai<2.6,>=2.3.1
        Downloading fastai-2.5.6-py3-none-any.whl (188 kB)
      [K     |████████████████████████████████| 188 kB 60.9 MB/s 
      [?25hCollecting xgboost<1.5,>=1.4
        Downloading xgboost-1.4.2-py3-none-manylinux2010_x86_64.whl (166.7 MB)
      [K     |████████████████████████████████| 166.7 MB 18 kB/s 
      [?25hCollecting lightgbm<3.4,>=3.3
        Downloading lightgbm-3.3.2-py3-none-manylinux1_x86_64.whl (2.0 MB)
      [K     |████████████████████████████████| 2.0 MB 39.8 MB/s 
      [?25hCollecting psutil<6,>=5.7.3
        Downloading psutil-5.8.0-cp37-cp37m-manylinux2010_x86_64.whl (296 kB)
      [K     |████████████████████████████████| 296 kB 75.4 MB/s 
      [?25hCollecting gluonts>=0.8.0
        Downloading gluonts-0.10.0-py3-none-any.whl (2.5 MB)
      [K     |████████████████████████████████| 2.5 MB 41.8 MB/s 
      [?25hCollecting sktime~=0.12
        Downloading sktime-0.12.1-py3-none-any.whl (6.8 MB)
      [K     |████████████████████████████████| 6.8 MB 48.0 MB/s 
      [?25hCollecting tbats~=1.1
        Downloading tbats-1.1.0-py3-none-any.whl (43 kB)
      [K     |████████████████████████████████| 43 kB 2.5 MB/s 
      [?25hCollecting pmdarima~=1.8
        Downloading pmdarima-1.8.5-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_24_x86_64.whl (1.4 MB)
      [K     |████████████████████████████████| 1.4 MB 50.2 MB/s 
      [?25hCollecting gluoncv<0.10.6,>=0.10.5
        Downloading gluoncv-0.10.5.post0-py2.py3-none-any.whl (1.3 MB)
      [K     |████████████████████████████████| 1.3 MB 54.0 MB/s 
      [?25hRequirement already satisfied: graphviz in /usr/local/lib/python3.7/dist-packages (from catboost<1.1,>=1.0->autogluon.tabular[all]==0.5.0->autogluon) (0.10.1)
      Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from catboost<1.1,>=1.0->autogluon.tabular[all]==0.5.0->autogluon) (1.15.0)
      Requirement already satisfied: plotly in /usr/local/lib/python3.7/dist-packages (from catboost<1.1,>=1.0->autogluon.tabular[all]==0.5.0->autogluon) (5.5.0)
      Collecting partd>=0.3.10
        Downloading partd-1.2.0-py3-none-any.whl (19 kB)
      Requirement already satisfied: pyyaml in /usr/local/lib/python3.7/dist-packages (from dask<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (3.13)
      Requirement already satisfied: cloudpickle>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from dask<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (1.3.0)
      Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from dask<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (21.3)
      Collecting fsspec>=0.6.0
        Downloading fsspec-2022.5.0-py3-none-any.whl (140 kB)
      [K     |████████████████████████████████| 140 kB 62.9 MB/s 
      [?25hRequirement already satisfied: toolz>=0.8.2 in /usr/local/lib/python3.7/dist-packages (from dask<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (0.11.2)
      Requirement already satisfied: tblib>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (1.7.0)
      Requirement already satisfied: zict>=0.1.3 in /usr/local/lib/python3.7/dist-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (2.2.0)
      Requirement already satisfied: sortedcontainers!=2.0.0,!=2.0.1 in /usr/local/lib/python3.7/dist-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (2.4.0)
      Requirement already satisfied: click>=6.6 in /usr/local/lib/python3.7/dist-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (7.1.2)
      Requirement already satisfied: tornado>=5 in /usr/local/lib/python3.7/dist-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (5.1.1)
      Requirement already satisfied: jinja2 in /usr/local/lib/python3.7/dist-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (2.11.3)
      Collecting cloudpickle>=1.1.1
        Downloading cloudpickle-2.1.0-py3-none-any.whl (25 kB)
      Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (57.4.0)
      Requirement already satisfied: msgpack>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (1.0.4)
      Requirement already satisfied: spacy<4 in /usr/local/lib/python3.7/dist-packages (from fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (3.3.1)
      Requirement already satisfied: fastcore<1.5,>=1.3.27 in /usr/local/lib/python3.7/dist-packages (from fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (1.4.4)
      Requirement already satisfied: fastprogress>=0.2.4 in /usr/local/lib/python3.7/dist-packages (from fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (1.0.2)
      Requirement already satisfied: fastdownload<2,>=0.0.5 in /usr/local/lib/python3.7/dist-packages (from fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (0.0.6)
      Requirement already satisfied: torchvision>=0.8.2 in /usr/local/lib/python3.7/dist-packages (from fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (0.12.0+cu113)
      Requirement already satisfied: pip in /usr/local/lib/python3.7/dist-packages (from fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (21.1.3)
      Requirement already satisfied: opencv-python in /usr/local/lib/python3.7/dist-packages (from gluoncv<0.10.6,>=0.10.5->autogluon.vision==0.5.0->autogluon) (4.1.2.30)
      Collecting autocfg
        Downloading autocfg-0.0.8-py3-none-any.whl (13 kB)
      Collecting portalocker
        Downloading portalocker-2.4.0-py2.py3-none-any.whl (16 kB)
      Requirement already satisfied: pydantic~=1.7 in /usr/local/lib/python3.7/dist-packages (from gluonts>=0.8.0->autogluon.timeseries[all]==0.5.0->autogluon) (1.8.2)
      Requirement already satisfied: typing-extensions~=4.0 in /usr/local/lib/python3.7/dist-packages (from gluonts>=0.8.0->autogluon.timeseries[all]==0.5.0->autogluon) (4.1.1)
      Requirement already satisfied: holidays>=0.9 in /usr/local/lib/python3.7/dist-packages (from gluonts>=0.8.0->autogluon.timeseries[all]==0.5.0->autogluon) (0.10.5.2)
      Requirement already satisfied: python-dateutil in /usr/local/lib/python3.7/dist-packages (from holidays>=0.9->gluonts>=0.8.0->autogluon.timeseries[all]==0.5.0->autogluon) (2.8.2)
      Requirement already satisfied: korean-lunar-calendar in /usr/local/lib/python3.7/dist-packages (from holidays>=0.9->gluonts>=0.8.0->autogluon.timeseries[all]==0.5.0->autogluon) (0.2.1)
      Requirement already satisfied: convertdate>=2.3.0 in /usr/local/lib/python3.7/dist-packages (from holidays>=0.9->gluonts>=0.8.0->autogluon.timeseries[all]==0.5.0->autogluon) (2.4.0)
      Requirement already satisfied: hijri-converter in /usr/local/lib/python3.7/dist-packages (from holidays>=0.9->gluonts>=0.8.0->autogluon.timeseries[all]==0.5.0->autogluon) (2.2.4)
      Requirement already satisfied: pymeeus<=1,>=0.3.13 in /usr/local/lib/python3.7/dist-packages (from convertdate>=2.3.0->holidays>=0.9->gluonts>=0.8.0->autogluon.timeseries[all]==0.5.0->autogluon) (0.5.11)
      Requirement already satisfied: future in /usr/local/lib/python3.7/dist-packages (from hyperopt<0.2.8,>=0.2.7->autogluon.core[all]==0.5.0->autogluon) (0.16.0)
      Collecting py4j
        Downloading py4j-0.10.9.5-py2.py3-none-any.whl (199 kB)
      [K     |████████████████████████████████| 199 kB 76.3 MB/s 
      [?25hRequirement already satisfied: wheel in /usr/local/lib/python3.7/dist-packages (from lightgbm<3.4,>=3.3->autogluon.tabular[all]==0.5.0->autogluon) (0.37.1)
      Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->autogluon.core[all]==0.5.0->autogluon) (0.11.0)
      Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->autogluon.core[all]==0.5.0->autogluon) (1.4.3)
      Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->autogluon.core[all]==0.5.0->autogluon) (3.0.9)
      Requirement already satisfied: gdown>=4.0.0 in /usr/local/lib/python3.7/dist-packages (from nlpaug<2.0.0,>=1.1.10->autogluon.multimodal==0.5.0->autogluon) (4.4.0)
      Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.7/dist-packages (from gdown>=4.0.0->nlpaug<2.0.0,>=1.1.10->autogluon.multimodal==0.5.0->autogluon) (4.6.3)
      Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from gdown>=4.0.0->nlpaug<2.0.0,>=1.1.10->autogluon.multimodal==0.5.0->autogluon) (3.7.1)
      Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (from nltk<4.0.0,>=3.4.5->autogluon.multimodal==0.5.0->autogluon) (1.1.0)
      Collecting typish>=1.7.0
        Downloading typish-1.9.3-py3-none-any.whl (45 kB)
      [K     |████████████████████████████████| 45 kB 3.7 MB/s 
      [?25hCollecting pyyaml
        Downloading PyYAML-6.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (596 kB)
      [K     |████████████████████████████████| 596 kB 50.5 MB/s 
      [?25hCollecting antlr4-python3-runtime==4.8
        Downloading antlr4-python3-runtime-4.8.tar.gz (112 kB)
      [K     |████████████████████████████████| 112 kB 59.2 MB/s 
      [?25hRequirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas!=1.4.0,<1.5,>=1.2.5->autogluon.core[all]==0.5.0->autogluon) (2022.1)
      Collecting locket
        Downloading locket-1.0.0-py2.py3-none-any.whl (4.4 kB)
      Collecting statsmodels!=0.12.0,>=0.11
        Downloading statsmodels-0.13.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (9.8 MB)
      [K     |████████████████████████████████| 9.8 MB 46.2 MB/s 
      [?25hRequirement already satisfied: Cython!=0.29.18,>=0.29 in /usr/local/lib/python3.7/dist-packages (from pmdarima~=1.8->autogluon.timeseries[all]==0.5.0->autogluon) (0.29.30)
      Requirement already satisfied: urllib3 in /usr/local/lib/python3.7/dist-packages (from pmdarima~=1.8->autogluon.timeseries[all]==0.5.0->autogluon) (1.24.3)
      Collecting pyDeprecate>=0.3.1
        Downloading pyDeprecate-0.3.2-py3-none-any.whl (10 kB)
      Requirement already satisfied: tensorboard>=2.2.0 in /usr/local/lib/python3.7/dist-packages (from pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (2.8.0)
      Collecting aiohttp
        Downloading aiohttp-3.8.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.1 MB)
      [K     |████████████████████████████████| 1.1 MB 40.2 MB/s 
      [?25hRequirement already satisfied: jsonschema in /usr/local/lib/python3.7/dist-packages (from ray<1.14,>=1.13->autogluon.core[all]==0.5.0->autogluon) (4.3.3)
      Requirement already satisfied: attrs in /usr/local/lib/python3.7/dist-packages (from ray<1.14,>=1.13->autogluon.core[all]==0.5.0->autogluon) (21.4.0)
      Collecting frozenlist
        Downloading frozenlist-1.3.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (144 kB)
      [K     |████████████████████████████████| 144 kB 47.5 MB/s 
      [?25hCollecting grpcio<=1.43.0,>=1.28.1
        Downloading grpcio-1.43.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.1 MB)
      [K     |████████████████████████████████| 4.1 MB 45.9 MB/s 
      [?25hCollecting aiosignal
        Downloading aiosignal-1.2.0-py3-none-any.whl (8.2 kB)
      Collecting virtualenv
        Downloading virtualenv-20.15.1-py2.py3-none-any.whl (10.1 MB)
      [K     |████████████████████████████████| 10.1 MB 51.3 MB/s 
      [?25hCollecting tensorboardX>=1.9
        Downloading tensorboardX-2.5.1-py2.py3-none-any.whl (125 kB)
      [K     |████████████████████████████████| 125 kB 73.3 MB/s 
      [?25hRequirement already satisfied: tabulate in /usr/local/lib/python3.7/dist-packages (from ray<1.14,>=1.13->autogluon.core[all]==0.5.0->autogluon) (0.8.9)
      Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->autogluon.core[all]==0.5.0->autogluon) (2022.6.15)
      Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->autogluon.core[all]==0.5.0->autogluon) (3.0.4)
      Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->autogluon.core[all]==0.5.0->autogluon) (2.10)
      Requirement already satisfied: imageio>=2.4.1 in /usr/local/lib/python3.7/dist-packages (from scikit-image<0.20.0,>=0.19.1->autogluon.multimodal==0.5.0->autogluon) (2.4.1)
      Requirement already satisfied: PyWavelets>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from scikit-image<0.20.0,>=0.19.1->autogluon.multimodal==0.5.0->autogluon) (1.3.0)
      Requirement already satisfied: tifffile>=2019.7.26 in /usr/local/lib/python3.7/dist-packages (from scikit-image<0.20.0,>=0.19.1->autogluon.multimodal==0.5.0->autogluon) (2021.11.2)
      Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn<1.1,>=1.0.0->autogluon.core[all]==0.5.0->autogluon) (3.1.0)
      Collecting deprecated>=1.2.13
        Downloading Deprecated-1.2.13-py2.py3-none-any.whl (9.6 kB)
      Collecting numba>=0.53
        Downloading numba-0.55.2-cp37-cp37m-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.3 MB)
      [K     |████████████████████████████████| 3.3 MB 32.6 MB/s 
      [?25hRequirement already satisfied: wrapt<2,>=1.10 in /usr/local/lib/python3.7/dist-packages (from deprecated>=1.2.13->sktime~=0.12->autogluon.timeseries[all]==0.5.0->autogluon) (1.14.1)
      Collecting llvmlite<0.39,>=0.38.0rc1
        Downloading llvmlite-0.38.1-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (34.5 MB)
      [K     |████████████████████████████████| 34.5 MB 16 kB/s 
      [?25hRequirement already satisfied: thinc<8.1.0,>=8.0.14 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (8.0.17)
      Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (1.0.7)
      Requirement already satisfied: langcodes<4.0.0,>=3.2.0 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (3.3.0)
      Requirement already satisfied: typer<0.5.0,>=0.3.0 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (0.4.1)
      Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (1.0.2)
      Requirement already satisfied: pathy>=0.3.5 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (0.6.1)
      Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.9 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (3.0.9)
      Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (3.0.6)
      Requirement already satisfied: blis<0.8.0,>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (0.7.7)
      Requirement already satisfied: catalogue<2.1.0,>=2.0.6 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (2.0.7)
      Requirement already satisfied: wasabi<1.1.0,>=0.9.1 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (0.9.1)
      Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (2.0.6)
      Requirement already satisfied: srsly<3.0.0,>=2.4.3 in /usr/local/lib/python3.7/dist-packages (from spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (2.4.3)
      Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from catalogue<2.1.0,>=2.0.6->spacy<4->fastai<2.6,>=2.3.1->autogluon.tabular[all]==0.5.0->autogluon) (3.8.0)
      Requirement already satisfied: patsy>=0.5.2 in /usr/local/lib/python3.7/dist-packages (from statsmodels!=0.12.0,>=0.11->pmdarima~=1.8->autogluon.timeseries[all]==0.5.0->autogluon) (0.5.2)
      Requirement already satisfied: absl-py>=0.4 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (1.1.0)
      Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (1.35.0)
      Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (0.6.1)
      Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (1.0.1)
      Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (1.8.1)
      Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (0.4.6)
      Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (3.3.7)
      Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (0.2.8)
      Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (4.2.4)
      Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (4.8)
      Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (1.3.1)
      Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (4.11.4)
      Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (0.4.8)
      Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard>=2.2.0->pytorch-lightning<1.7.0,>=1.5.10->autogluon.multimodal==0.5.0->autogluon) (3.2.0)
      Collecting huggingface-hub<1.0,>=0.1.0
        Downloading huggingface_hub-0.8.1-py3-none-any.whl (101 kB)
      [K     |████████████████████████████████| 101 kB 12.7 MB/s 
      [?25hRequirement already satisfied: heapdict in /usr/local/lib/python3.7/dist-packages (from zict>=0.1.3->distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (1.0.1)
      Requirement already satisfied: charset-normalizer<3.0,>=2.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->fsspec>=0.6.0->dask<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (2.0.12)
      Collecting yarl<2.0,>=1.0
        Downloading yarl-1.7.2-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (271 kB)
      [K     |████████████████████████████████| 271 kB 60.2 MB/s 
      [?25hCollecting asynctest==0.13.0
        Downloading asynctest-0.13.0-py3-none-any.whl (26 kB)
      Collecting async-timeout<5.0,>=4.0.0a3
        Downloading async_timeout-4.0.2-py3-none-any.whl (5.8 kB)
      Collecting multidict<7.0,>=4.5
        Downloading multidict-6.0.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (94 kB)
      [K     |████████████████████████████████| 94 kB 4.3 MB/s 
      [?25hCollecting jmespath<2.0.0,>=0.7.1
        Downloading jmespath-1.0.1-py3-none-any.whl (20 kB)
      Collecting s3transfer<0.7.0,>=0.6.0
        Downloading s3transfer-0.6.0-py3-none-any.whl (79 kB)
      [K     |████████████████████████████████| 79 kB 9.8 MB/s 
      [?25hCollecting botocore<1.28.0,>=1.27.24
        Downloading botocore-1.27.24-py3-none-any.whl (9.0 MB)
      [K     |████████████████████████████████| 9.0 MB 42.8 MB/s 
      [?25hCollecting urllib3
        Downloading urllib3-1.25.11-py2.py3-none-any.whl (127 kB)
      [K     |████████████████████████████████| 127 kB 77.5 MB/s 
      [?25hCollecting immutables>=0.9
        Downloading immutables-0.18-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (116 kB)
      [K     |████████████████████████████████| 116 kB 64.3 MB/s 
      [?25hCollecting mccabe<0.7.0,>=0.6.0
        Downloading mccabe-0.6.1-py2.py3-none-any.whl (8.6 kB)
      Collecting flake8
        Downloading flake8-4.0.0-py2.py3-none-any.whl (64 kB)
      [K     |████████████████████████████████| 64 kB 3.3 MB/s 
      [?25h  Downloading flake8-3.9.2-py2.py3-none-any.whl (73 kB)
      [K     |████████████████████████████████| 73 kB 2.2 MB/s 
      [?25hCollecting pycodestyle<2.8.0,>=2.7.0
        Downloading pycodestyle-2.7.0-py2.py3-none-any.whl (41 kB)
      [K     |████████████████████████████████| 41 kB 677 kB/s 
      [?25hCollecting pyflakes<2.4.0,>=2.3.0
        Downloading pyflakes-2.3.1-py2.py3-none-any.whl (68 kB)
      [K     |████████████████████████████████| 68 kB 7.9 MB/s 
      [?25hRequirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.7/dist-packages (from jinja2->distributed<=2021.11.2,>=2021.09.1->autogluon.core[all]==0.5.0->autogluon) (2.0.1)
      Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema->ray<1.14,>=1.13->autogluon.core[all]==0.5.0->autogluon) (0.18.1)
      Requirement already satisfied: importlib-resources>=1.4.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema->ray<1.14,>=1.13->autogluon.core[all]==0.5.0->autogluon) (5.7.1)
      Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.7/dist-packages (from plotly->catboost<1.1,>=1.0->autogluon.tabular[all]==0.5.0->autogluon) (8.0.1)
      Requirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /usr/local/lib/python3.7/dist-packages (from requests->autogluon.core[all]==0.5.0->autogluon) (1.7.1)
      Collecting colorama
        Downloading colorama-0.4.5-py2.py3-none-any.whl (16 kB)
      Collecting platformdirs<3,>=2
        Downloading platformdirs-2.5.2-py3-none-any.whl (14 kB)
      Collecting distlib<1,>=0.3.1
        Downloading distlib-0.3.4-py2.py3-none-any.whl (461 kB)
      [K     |████████████████████████████████| 461 kB 48.1 MB/s 
      [?25hBuilding wheels for collected packages: fairscale, antlr4-python3-runtime, sacremoses, contextvars
        Building wheel for fairscale (PEP 517) ... [?25l[?25hdone
        Created wheel for fairscale: filename=fairscale-0.4.6-py3-none-any.whl size=307252 sha256=0b2c65e6e7c9a2f136d8a5a7eadcb1e99c8cfeb6356ca2cbad7827101dcdd117
        Stored in directory: /root/.cache/pip/wheels/4e/4f/0b/94c29ea06dfad93260cb0377855f87b7b863312317a7f69fe7
        Building wheel for antlr4-python3-runtime (setup.py) ... [?25l[?25hdone
        Created wheel for antlr4-python3-runtime: filename=antlr4_python3_runtime-4.8-py3-none-any.whl size=141230 sha256=e29ca1f4489f2ca60db58e4067569515a20cb097e927eed0557045c286f4950c
        Stored in directory: /root/.cache/pip/wheels/ca/33/b7/336836125fc9bb4ceaa4376d8abca10ca8bc84ddc824baea6c
        Building wheel for sacremoses (setup.py) ... [?25l[?25hdone
        Created wheel for sacremoses: filename=sacremoses-0.0.53-py3-none-any.whl size=895260 sha256=5cd83f92e2b3a7b58cc592b658ecd9248d9cffd0ba41f987f53f1050b8218209
        Stored in directory: /root/.cache/pip/wheels/87/39/dd/a83eeef36d0bf98e7a4d1933a4ad2d660295a40613079bafc9
        Building wheel for contextvars (setup.py) ... [?25l[?25hdone
        Created wheel for contextvars: filename=contextvars-2.4-py3-none-any.whl size=7681 sha256=a9817a932a59e682f94686b12e586893ca60c4b2287c2e6bc471bb66de870c1c
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

The dataset we will use contains individuals’ information such as
occupation with if or not her income exceeds $50,000, which is the
predicting target. We load this dataset directly from a URL. Note that
the ``TabularDataset`` class is a subclass of `pandas
DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`__,
any pandas methods can be applied here.

.. code:: python

    url = 'https://autogluon.s3.amazonaws.com/datasets/Inc/'
    train_data = TabularDataset(url+'train.csv')
    train_data.head()




.. raw:: html

    
      <div id="df-378bceb4-a99f-4c6d-ba59-d2b233aca851">
        <div class="colab-df-container table-wrapper">
          <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table class="dataframe docutils">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>age</th>
          <th>workclass</th>
          <th>fnlwgt</th>
          <th>education</th>
          <th>education-num</th>
          <th>marital-status</th>
          <th>occupation</th>
          <th>relationship</th>
          <th>race</th>
          <th>sex</th>
          <th>capital-gain</th>
          <th>capital-loss</th>
          <th>hours-per-week</th>
          <th>native-country</th>
          <th>class</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>25</td>
          <td>Private</td>
          <td>178478</td>
          <td>Bachelors</td>
          <td>13</td>
          <td>Never-married</td>
          <td>Tech-support</td>
          <td>Own-child</td>
          <td>White</td>
          <td>Female</td>
          <td>0</td>
          <td>0</td>
          <td>40</td>
          <td>United-States</td>
          <td>&lt;=50K</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23</td>
          <td>State-gov</td>
          <td>61743</td>
          <td>5th-6th</td>
          <td>3</td>
          <td>Never-married</td>
          <td>Transport-moving</td>
          <td>Not-in-family</td>
          <td>White</td>
          <td>Male</td>
          <td>0</td>
          <td>0</td>
          <td>35</td>
          <td>United-States</td>
          <td>&lt;=50K</td>
        </tr>
        <tr>
          <th>2</th>
          <td>46</td>
          <td>Private</td>
          <td>376789</td>
          <td>HS-grad</td>
          <td>9</td>
          <td>Never-married</td>
          <td>Other-service</td>
          <td>Not-in-family</td>
          <td>White</td>
          <td>Male</td>
          <td>0</td>
          <td>0</td>
          <td>15</td>
          <td>United-States</td>
          <td>&lt;=50K</td>
        </tr>
        <tr>
          <th>3</th>
          <td>55</td>
          <td>?</td>
          <td>200235</td>
          <td>HS-grad</td>
          <td>9</td>
          <td>Married-civ-spouse</td>
          <td>?</td>
          <td>Husband</td>
          <td>White</td>
          <td>Male</td>
          <td>0</td>
          <td>0</td>
          <td>50</td>
          <td>United-States</td>
          <td>&gt;50K</td>
        </tr>
        <tr>
          <th>4</th>
          <td>36</td>
          <td>Private</td>
          <td>224541</td>
          <td>7th-8th</td>
          <td>4</td>
          <td>Married-civ-spouse</td>
          <td>Handlers-cleaners</td>
          <td>Husband</td>
          <td>White</td>
          <td>Male</td>
          <td>0</td>
          <td>0</td>
          <td>40</td>
          <td>El-Salvador</td>
          <td>&lt;=50K</td>
        </tr>
      </tbody>
    </table>
    </div>
          <button class="colab-df-convert" onclick="convertToInteractive('df-378bceb4-a99f-4c6d-ba59-d2b233aca851')"
                  title="Convert this dataframe to an interactive table."
                  style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
           width="24px">
        <path d="M0 0h24v24H0V0z" fill="none"/>
        <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
      </svg>
          </button>
    
      <style>
        .colab-df-container {
          display:flex;
          flex-wrap:wrap;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
          <script>
            const buttonEl =
              document.querySelector('#df-378bceb4-a99f-4c6d-ba59-d2b233aca851 button.colab-df-convert');
            buttonEl.style.display =
              google.colab.kernel.accessAllowed ? 'block' : 'none';
    
            async function convertToInteractive(key) {
              const element = document.querySelector('#df-378bceb4-a99f-4c6d-ba59-d2b233aca851');
              const dataTable =
                await google.colab.kernel.invokeFunction('convertToInteractive',
                                                         [key], {});
              if (!dataTable) return;
    
              const docLinkHtml = 'Like what you see? Visit the ' +
                '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
                + ' to learn more about interactive tables.';
              element.innerHTML = '';
              dataTable['output_type'] = 'display_data';
              await google.colab.output.renderOutput(dataTable, element);
              const docLink = document.createElement('div');
              docLink.innerHTML = docLinkHtml;
              element.appendChild(docLink);
            }
          </script>
        </div>
      </div>




Our targets are stored in the ``class`` column, which has two unique
values.

.. code:: python

    label = 'class'
    train_data[label].describe()




.. parsed-literal::

    count      39073
    unique         2
    top        <=50K
    freq       29704
    Name: class, dtype: object



Now construct a ``TabularPredictor`` instance by specifying the label
column name, then train on the dataset with the
:func:`autogluon.tabular.TabularPredictor.fit` method. We don’t need
to specify any other hyperparameters. This method will perform automatic
feature engineering, train multiple models, and then ensemble them to
form the final predictions. You can find detailed information in the
output log.

.. code:: python

   predictor = TabularPredictor(label=label).fit(train_data)

.. container:: {toggle}

   ::

      No path specified. Models will be saved in: "AutogluonModels/ag-20220707_055423/"
      Beginning AutoGluon training ...
      AutoGluon will save models to "AutogluonModels/ag-20220707_055423/"
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
          Available Memory:                    12688.8 MB
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
          0.5s = Fit runtime
          14 features in original data used to generate 14 features in processed data.
          Train Data (Processed) Memory Usage: 2.19 MB (0.0% of available memory)
      Data preprocessing and feature engineering runtime = 0.65s ...
      AutoGluon will gauge predictive performance using evaluation metric: 'accuracy'
          To change this, specify the eval_metric parameter of Predictor()
      Automatically generating train/validation split with holdout_frac=0.0639828014229775, Train Rows: 36573, Val Rows: 2500
      Fitting 13 L1 models ...
      Fitting model: KNeighborsUnif ...
          0.7752   = Validation score   (accuracy)
          0.08s    = Training   runtime
          0.13s    = Validation runtime
      Fitting model: KNeighborsDist ...
          0.766    = Validation score   (accuracy)
          0.07s    = Training   runtime
          0.11s    = Validation runtime
      Fitting model: LightGBMXT ...
          0.8792   = Validation score   (accuracy)
          5.5s     = Training   runtime
          0.12s    = Validation runtime
      Fitting model: LightGBM ...
          0.8824   = Validation score   (accuracy)
          3.57s    = Training   runtime
          0.13s    = Validation runtime
      Fitting model: RandomForestGini ...
          0.8588   = Validation score   (accuracy)
          18.35s   = Training   runtime
          0.31s    = Validation runtime
      Fitting model: RandomForestEntr ...
          0.862    = Validation score   (accuracy)
          12.39s   = Training   runtime
          0.31s    = Validation runtime
      Fitting model: CatBoost ...
          0.8836   = Validation score   (accuracy)
          33.68s   = Training   runtime
          0.02s    = Validation runtime
      Fitting model: ExtraTreesGini ...
          0.85     = Validation score   (accuracy)
          6.56s    = Training   runtime
          0.31s    = Validation runtime
      Fitting model: ExtraTreesEntr ...
          0.848    = Validation score   (accuracy)
          6.97s    = Training   runtime
          0.31s    = Validation runtime
      Fitting model: NeuralNetFastAI ...
          0.864    = Validation score   (accuracy)
          46.39s   = Training   runtime
          0.05s    = Validation runtime
      Fitting model: XGBoost ...
          0.8872   = Validation score   (accuracy)
          2.02s    = Training   runtime
          0.03s    = Validation runtime
      Fitting model: NeuralNetTorch ...
          0.8584   = Validation score   (accuracy)
          46.86s   = Training   runtime
          0.06s    = Validation runtime
      Fitting model: LightGBMLarge ...
          0.8856   = Validation score   (accuracy)
          2.72s    = Training   runtime
          0.09s    = Validation runtime
      Fitting model: WeightedEnsemble_L2 ...
          0.8912   = Validation score   (accuracy)
          1.39s    = Training   runtime
          0.0s     = Validation runtime
      AutoGluon training complete, total runtime = 195.61s ... Best model: "WeightedEnsemble_L2"
      TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220707_055423/")

The training is often fast, as in default AutoGluon will not try very
big models. For example, training the above dataset with ~40K rows
should be finished within a few minutes on a normal CPU. But if you want
AutoGluon to stop earlier, you can specify the ``time_limit`` argument
in the ``fit`` method. For example, ``fit(..., time_limit=60, ...)``
means training at most 1 minute.

Once training is done, load separate test data to predict.

.. code:: python

    test_data = TabularDataset(url+'test.csv')
    # Optional: delete the label column for safety check.
    y_pred = predictor.predict(test_data.drop(columns=[label]))
    y_pred.head()


.. parsed-literal::

    Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769




.. parsed-literal::

    0     <=50K
    1     <=50K
    2      >50K
    3     <=50K
    4     <=50K
    Name: class, dtype: object



If you just want to evaluate the model performance, you can call the
:func:`autogluon.tabular.TabularPredictor.evaluate` method.

.. code:: python

    predictor.evaluate(test_data, silent=True)




.. parsed-literal::

    {'accuracy': 0.8763435356740711,
     'balanced_accuracy': 0.7950062351568354,
     'f1': 0.710727969348659,
     'mcc': 0.6395678748952276,
     'precision': 0.798708288482239,
     'recall': 0.640207075064711,
     'roc_auc': 0.9313343583022541}



Now we did a quick through about using AutoGluon for tabular prediction.
We used two classes, :class:`autogluon.tabular.TabularDataset`
(essentially a pandas DataFrame) to load data and
:class:`autogluon.tabular.TabularPredictor` to train (via the
``fit`` method) and predict (via the ``predict`` method). You will see
similar APIs for other tasks, namely a ``Dataset`` class to load data
and a ``Prediction`` class to train and predict.

In addition, AutoGluon simplifies the model training by not requiring
feature engineering and specifying model hyperparameters. AutoGluon
automatically performs these jobs when running ``fit``. You may worry
about the resulted longer training time, AutoGluon balances the
computational cost and model quality. You can benchmark AutoGluon’s
performance on the whole dataset loaded above against your favoriate
machine learning model. But to be fair, you also need to count the time
you spend on preprocessing data and tuning your models.

To know more about AutoGluon, next you can read

-  the cheetsheet for a quick overview of the APIs
-  tutorials to customize the training and inference
-  understand how AutoGluon performs feature engineering and model
   ensemble.
