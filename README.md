# Postprocessing Ensemble Forecasts with Generative Adversarial Network (PEFGAN)


### Our model is GAN-based and includes an ensemble forecast metric CRPS (Continuous Ranked Probability Score) into loss function to learn from a whole of ensemble forecasts to improve ensemble and extreme forecasts. 
PEFGAN consistently makes better forecasts measured by CPRS, Brier score for the 99th percentile precipitation, MAE, and Relative Bias than QM, DESRGAN and long-standing SCF(Seasonal Climate Forecasts) benchmark climatology, respectively. 

![Language](https://img.shields.io/badge/language-python3-informational)



---

In this project, the whole progress can be divided into 4 stages:<br>
&emsp;&emsp;1.&emsp;Data pre-processing and masking<br>
&emsp;&emsp;2.&emsp;Training<br>
&emsp;&emsp;3.&emsp;Testing and evaluation<br>
&emsp;&emsp;4.&emsp;Visualisation<br>

#### Data pre-processing and masking

***

&emsp;In this step, our program will read the **ACCESS-S2** or **QM** data from **gadi** and crop the desired region which will be used for train. Due to the fact that we only care about the precipitation on the mainland rather than the rainfall in the ocean, we also need to mask the ocean area. More detailed explanation, you can find a file called **instruction.txt** in folder **Data_process_code**.

#### Training

***

&emsp; Trainning code is located in **PEFGAN** folder, which include the **train.py**, and **utils.py** with some comments of code. There also has code related to the net architecture. 

#### Testing and evaluation

***

&emsp; Testing code is also located in **PEFGAN** folder, which include the **test.py**, and **eval_PEFGAN.py**. 

#### Visualisation

***

| ![Watch the video](/image/44p.gif) | 
|:--:| 
| *Visual comparison on ACCESS-S2, QM, DESRGAN output, PEFGAN output, target date.* |


### WorkFlow/Pipeline

***

&emsp;&emsp;1.&emsp;Run 

```
ACCESS_e*.py

QM_data_crop_e*.py

agcd_mask_processing.py 
```
in folder **Data_process_code** to crop desired area using latitude and longitude and mask the ocean. <br>

&emsp;&emsp;2.&emsp;Run 
```
train.py

test.py 

eval_PEFGAN.py 
```
sequentially in **PEFGAN**.  <be>

&emsp;&emsp;3.&emsp;After you ran eval_PEFGAN.py, you will get the CRPS of your model and you can continue to run the corresponding climitology.py for a specific year in folder **CRPS calculation code**.<be>

&emsp;&emsp;4.&emsp;Then run table_csv.py to calculate the mean CRPS of climatology data for each leading time date of that year.<be>

&emsp;&emsp;5.&emsp;Finally you can run csv_crps_ss.py to calculate CRPS skill score, MAE, Relative Bias, Brier score for your own model(PEFGAN), DESRGAN, QM or so and output a csv file. The CRPS images for each leading time will be stored in the desired folder based on your saving path.
