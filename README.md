# capstone_group2

## WorldQuant University MSc Financial Engineering Capstone Project - Group 2

## Project Title : 
    Statistical Characteristics of Markets: 

    The Case between Developed and Emerging Equity Markets


Data duration is from June 2006 to May 2020


Analysis in these periods

* Full Time Period: June 2006 – May 2020

* 2008 Financial Crisis Period: July 2007 – June 2009

* In between: July 2009 – Nov. 2019

* Covid-19 Period: Dec. 2019 – May 2020


Market Indices used are : 
* S&P 500 (representing the developed market),

* JSE.JO (representing South Africa emerging market) 

* 000001.SS Shanghai Stock Exchange Composite Index (representing China emerging market)

Use cases included in this code :
* Download the data for indices into excel
* Download FX data for ZAR-USD and CNY-USD into excel 
* Convert the prices for JSE from ZAR to USD in excel
* Convert the prices for 000001.SS from CNY to USD in excel
* Calulate returns and volatility with plots 
* Describe the basic statistics of the data
* Normality tests of stock returns together with plots
* Correlation between stock prices and returns
* Test for auto-correlation and plots
* Stationarity Tests for returns 
* Volatility Modelling for the Market Returns (ARCH and GARCH models) 
* Long and Short run relationship between S&P 500 and the other 2 
* Regime Switching Model using the Markov's Regression model for 3 and 2 regimes


## Content of this project. 
    This project was orginally done in Jupyter-Notebook; an in-browser based IDLE for python projects.

It contains the main python script named main.py, an excel file with adjusted close prices and another excel file with the entire historical data, FX data and conversion data for the currencies. 

## DATA
The data used are from 3 different countries in 3 different continents with 3 different base currencies. 

In The data preparation stage, the foreign exchange rates data (daily rates) was used to convert the SSE Limited and JSE Limited data to USD.

Since the S&P 500 is already in the USD, it was left intact. Hence all prices are in USD.

Further, the Adjusted Close Prices were isolated for the main analysis. 

## Details
For the purpose of submitting this project as part of the requirements for the WQU MScFE program, the repository is private until further revision. 

Create a Virtual environment in python by running the following on Windows

* pip install virtualenv virtualenvwrapper

* virtualenv *<name_of_your_virtual_environment>*

* *<name_of_your_virtual_enrionment>*\Scripts\activate 


Congratulations!!! 

Your vritual environment is ready for coding now. To deactivate the virtual environment, just run *deactivate*.

You can visit the python official documentations to read more about virtualenv and how to create and use it on other systems like Linux and MacOS. Here is a link to guide you. https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/ 


#### Cloning the code. 
    In your virtual environment, run the following command to clone this project.

    * git clone https://github.com/TMCreme/capstone_group2.git *
    
    This command downloads the entire project onto your local machine 
    
    Run the following command to enter the project's root directory
    
    * cd capstone_group2 * 

#### Running the code
To run the code, be sure that the Excel file containing the data is in the directory with the main.py file and run the following command
    
To install the modules used in the project

* *pip install -r requirements.txt* 

To exceute the code

* *python main.py*



### ACKNOWLEDGEMENT
* https://wqu.org 
* https://finance.yahoo.com
* https://www.investing.com
* https://python.org 


