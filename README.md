AIoT Course Project 2+ — Human Activity Recognition on PAMAP2

PAMAP2 IMU recordings are ingested into MongoDB and then fed to two
parallel ML solutions

Requirements

Python 3.11+
MongoDB Community Server 7.x

Setup

1. Download the PAMAP2 dataset from the UCI repository
<https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring>
and extract it so the project directory layout looks like:

   data/
   └── PAMAP2_Dataset/
       ├── Protocol/ 
       └── Optional/  

Do not rename or restructure the PAMAP2_Dataset/ directory.

2. Install MongoDB Community Server

3. Install Python
py -3.11 -m pip install -r requirements.txt


4.Copy config.yml.template to config.yml and adjust the values.

Execution order

Run the notebooks in this order.

1. aiot_dataset_creation.ipynb 
2. aiot_project_time_series.ipynb 
3. aiot_project_feature_engineering.ipynb 


Project structure

├── README.md                                   
├── requirements.txt                            
├── config.yml / config.yml.template            
├── utils.py                                    
├── utils_visual.py                             
├── aiot_dataset_creation.ipynb                
├── aiot_project_time_series.ipynb             
├── aiot_project_feature_engineering.ipynb     
├── img/                                        
├── data/
│   └── PAMAP2_Dataset/                         
└── mongodump_har/                              
