# Data description

In this README, I provide a description of the datasets included in DeepSurvK. These datasets are the ones that were used in [DeepSurv's original paper](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1) and were obtained directly from the [original repository](https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data).

Notice that if data are loaded as a NumPy array, you will only get the numerical values. In this case, the order of the columns correspond to the order of the variables described here. If data are loaded as a pandas DataFrame, you will also get the column headers.

:warning: [There might be some inconsistencies](https://github.com/jaredleekatzman/DeepSurv/issues/55) between the data reported in the original papers and the data included in DeepSurv (and therefore here). Moreover, these data consisted originally of only the numeric values. This made it hard to tell which values correspond to which feature. I obtained the variable names after researching the original sources of the data, as well as other studies that have been published using these datasets. I did my best, but unfortunately I cannot guarantee that these are 100% correct. For more information, please take a look at the suggested references.


## `metabric.h5`
| Variable name | Description | Units | Possible values  |
|---------------|-------------|-------|------------------|
| `gene1` | Gene 1 | ? | Continuous             |
| `gene2` | Gene 2 | ? | Continuous             |
| `gene3` | Gene 3 | ? | Continuous             |
| `gene4` | Gene 4 | ? | Continuous             |
| `radio` | Radiotherapy | - | 0 - No <br> 1 - Yes             |
| `horm` | Hormonal treatment | - | 0 - No <br> 1 - Yes             |
| `chemo` | Chemotherapy | - | 0 - No <br> 1 - Yes             |
| `er_status`    | Estrogen receptor status  | - | 0 - Negative <br> 1 - Positive             |
| `age`         | Age at diagnosis        | [years]| Continuous                |

:warning: `gene1`, `gene2`, `gene3`, and `gene4` presumably correspond to gene indicators or MKI67, EGFR, PGR, and ERBB2. However, it is hard to tell which one belongs to which. Therefore, their names here are kept generic.

References:
* Curtis, Christina, et al. ["The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups."](https://www.nature.com/articles/nature10983?__hstc=12316075.cde2cb5f07430159d50a3c91e72c280a.1492646400108.1492646400109.1492646400110.1&__hssc=12316075.1.1492646400111&__hsfp=1773666937) Nature 486.7403 (2012): 346-352.
* [METABRIC Data for Use in Independent Research](https://www.synapse.org/#!Synapse:syn1688369/wiki/27311)
* Obradovic and Silverman. ["Breast Cancer Treatment Outcomes Modeling from METABRIC Genomic Patient Data"](https://github.com/carsilverman/METABRIC_breast_cancer/blob/master/G4006_Final_Paper.pdf) (2018)

## `rgbsg.h5`

| Variable name | Description | Units | Possible values  |
|---------------|-------------|-------|------------------|
| `horm`        | Hormonal treatment | - | 0 - No <br> 1 - Yes             |
| `grade`       | Tumor grade        | - | 0 - Grade 1 <br> 1 - Grade 2 <br> 2 - Grade 3 |
| `menopause`   | Menopausal status  | - | 0 - Premenopausal <br> 1 - Postmenopausal             |
| `age`         | Age at diagnosis   | [years]| Continuous                |
| `n_positive_nodes` | No. of positive lymph nodes | - |              |
| `progesterone` | Concentration of progesterone receptor | [fmol/mg] |              |
| `estrogene`    | Concentration of estrogen receptor  | [fmol/mg]|             |

References:
* Schumacher, M., et al. ["Randomized 2 x 2 trial evaluating hormonal treatment and the duration of chemotherapy in node-positive breast cancer patients. German Breast Cancer Study Group."](https://ascopubs.org/doi/abs/10.1200/jco.1994.12.10.2086) Journal of Clinical Oncology 12.10 (1994): 2086-2093.
* [GBSG: German Breast Cancer Study Group](https://rdrr.io/cran/CoxRidge/man/GBSG.html) in CoxRidge (Cox Models with Dynamic Range Penalties)

## `simulated_gaussian.h5`

| Variable name | Description | Units | Possible values  |
|---------------|-------------|-------|------------------|
| `x1`          | Feature 1   | -     | Continuous       |
| `x2`          | Feature 2   | -     | Continuous       |
| `x3`          | Feature 3   | -     | Continuous       |
| `x4`          | Feature 4   | -     | Continuous       |
| `x5`          | Feature 5   | -     | Continuous       |
| `x6`          | Feature 6   | -     | Continuous       |
| `x7`          | Feature 7   | -     | Continuous       |
| `x8`          | Feature 8   | -     | Continuous       |
| `x9`          | Feature 9   | -     | Continuous       |
| `x10`         | Feature 10  | -     | Continuous       |

## `simulated_linear.h5`

| Variable name | Description | Units | Possible values  |
|---------------|-------------|-------|------------------|
| `x1`          | Feature 1   | -     | Continuous       |
| `x2`          | Feature 2   | -     | Continuous       |
| `x3`          | Feature 3   | -     | Continuous       |
| `x4`          | Feature 4   | -     | Continuous       |
| `x5`          | Feature 5   | -     | Continuous       |
| `x6`          | Feature 6   | -     | Continuous       |
| `x7`          | Feature 7   | -     | Continuous       |
| `x8`          | Feature 8   | -     | Continuous       |
| `x9`          | Feature 9   | -     | Continuous       |
| `x10`         | Feature 10  | -     | Continuous       |

## `simulated_treatment.h5`
Data similar to `simulated_gaussian` plus an additional `treatment` column.

| Variable name | Description | Units | Possible values  |
|---------------|-------------|-------|------------------|
| `x1`          | Feature 1   | -     | Continuous       |
| `x2`          | Feature 2   | -     | Continuous       |
| `x3`          | Feature 3   | -     | Continuous       |
| `x4`          | Feature 4   | -     | Continuous       |
| `x5`          | Feature 5   | -     | Continuous       |
| `x6`          | Feature 6   | -     | Continuous       |
| `x7`          | Feature 7   | -     | Continuous       |
| `x8`          | Feature 8   | -     | Continuous       |
| `x9`          | Feature 9   | -     | Continuous       |
| `x10`         | Feature 10  | -     | Continuous       |
| `treatment`   | Treatment   | -     | 0 - No <br> 1 - Yes |

## `support.h5`

| Variable name | Description | Units | Possible values  |
|---------------|-------------|-------|------------------|
| `age`         | Age at diagnosis        | [years]| Continuous                |
| `sex`         | Sex       | - | 0 - ? <br> 1 - ?             |
| `race`        | Race      | - | 0 - ? <br> 1 - ? <br> 2 - ? <br> 3 - ? <br> 4 - ? <br> 5 - ? <br> 6 - ? <br> 7 - ? <br> 8 - ? <br> 9 - ? |
| `n_comorbidities`         | Number of comorbidities | - | - |
| `diabetes`    | Patient suffers from diabetes | - | 0 - No <br> 1 - Yes |
| `dementia`    | Patient suffers from dementia | - | 0 - No <br> 1 - Yes |
| `cancer`      | Patient suffers from lung or colon cancer | - | 0 - No <br> 1 - Non-metastatic <br>  2 - Metastatic |
| `blood_pressure`      | Mean arterial blood pressure | mmHg | Continuous |
| `heart_rate`  | Heart rate | Breaths per minute (or Hz) | Continuous |
| `respiration_rate`  | Respiration rate | Breaths per minute (or Hz) | Continuous |
| `temperature`  | Temperature | Celsius degrees | Continuous |
| `white_blood_cell`  | White blood cell count | ? | Continuous |
| `serum_sodium`  | Sodium concentration in serum | mg/dL | Continuous |
| `serum_creatinine` | Creatinine concentration in serum | mg/dL | Continuous |

References:
* Knaus, William A., et al. ["The SUPPORT prognostic model: Objective estimates of survival for seriously ill hospitalized adults."](https://www.acpjournals.org/doi/abs/10.7326/0003-4819-122-3-199502010-00007) Annals of internal medicine 122.3 (1995): 191-203.


## `whas.h5`

| Variable name | Description | Units | Possible values  |
|---------------|-------------|-------|------------------|
| `shock`       | Cardiogenic shock as a complication during hospitalization | - | 0 - No <br> 1 - Yes |
| `age`         | Age at hospital admission        | [years]| Continuous |
| `sex`         | Sex       | - | 0 - Male <br> 1 - Female |
| `bmi`         | Body mass index       | [kg/m2]| Continuous |
| `chf`         | Congestive heart failure as a complication during hospitalization | - | 0 - No <br> 1 - Yes |
| `recurrent`   | Event recurrence (sometimes reported as `MIORD`) | - | 0 - No (first incidence) <br> 1 - Yes |

:warning: [Kvamme et al.](https://www.jmlr.org/papers/volume20/18-424/18-424.pdf) report that DeepSurv's version of the data (and therefore also this one) "is actually a case-control data set, meaning it contains multiple replications of some individuals". Apparently this was overlooked originally. I include this dataset for the sake of completion, but keep this in mind if you are using it.

References:
* Hosmer Jr, David W., Stanley Lemeshow, and Susanne May. [Applied survival analysis: regression modeling of time-to-event data.](https://edisciplinas.usp.br/pluginfile.php/1950055/mod_folder/content/0/Hosmer%20D.W.%2C%20Lemeshow%20S.%20Applied%20Survival%20Analysis.pdf?forcedownload=1) Vol. 618. John Wiley & Sons, 2011.
* Goldberg, Robert J., et al. ["Recent changes in attack and survival rates of acute myocardial infarction (1975 through 1981): the Worcester Heart Attack Study."](https://jamanetwork.com/journals/jama/article-abstract/404347) Jama 255.20 (1986): 2774-2779.
* Pagley, Paul R., et al. ["Gender differences in the treatment of patients with acute myocardial infarction: a multihospital, community-based perspective."](https://jamanetwork.com/journals/jamainternalmedicine/article-abstract/617114) Archives of internal medicine 153.5 (1993): 625-629.
