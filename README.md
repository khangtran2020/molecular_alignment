# Molecular Alignment 

* Author: Khang Tran - NJIT 
* Adviser: Loan Phan - HCMUE

## 1. Project Description:

In this project, we simulate the molecular alignment process using intense near resonant 
and non-adiabatic laser. This codes take in the laser's parameters and the molecules's parameters,
then output the distribution density on the polar angle between molecules and the laser. In addition,
the codes also output the degree of alignment `<cos^2 theta>` at these distribution. This value
lies between `(0,1)`, the closer it gets to `1`, the better the alignment process: 
* <cos^2 theta> == 0: perpendicular alignment
* <cos^2 theta> == 1/3: equally alignment
* <cos^2 theta> == 1: parallel alignment

## 2. Requirements:

* numpy
* pandas 
* matplotlib
* scipy
* json

## 3. Running the code:

* Change the parameters in the `input.json` file.
* Run `pip install -r requirements` in order to install the requirements packages
* Run `python mole_align.py` to run the alignment process.

## 4. Contacts:

* Khang Tran: `kt36@njit.edu`
* Loan Phan: `loanptn@hcmue.edu.vn`
