# HMM-LSTM Fusion Model for Economic Forecasting

### Python Version = 3.11.7

The paper can be found in the pdf file in the repo. To replicate the results the code has been provided.
The main_loaded.ipynb loads the saved models instead of training them from scratch. To ensure no errors, 
cloning the repository is not sufficient since it messes with the .pb files in the saved Models.
Instead download the zip file of the repo to ensure smooth process.

```
pip install -r requirements.txt
```

install tensorflow_macos==2.15.0 if using mac

To fetch the new data using the data_loader.py file a valid api key is needed from FRED. That can be retrieved from
https://fred.stlouisfed.org/docs/api/api_key.html

*For some reason the HMM produces different results on google colab.* 

If you run into a maximum file path limitation error on windows, please follow the instructions in this link
https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry

Ensure that if the powershell script is used, the powershell console is opened with admin privileges. 
