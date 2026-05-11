This project focuses on forecasting groundwater levels using a hybrid deep learning approach that combines:

PCA (Principal Component Analysis) for dimensionality reduction
DWT (Discrete Wavelet Transform) for signal denoising
LSTM (Long Short-Term Memory) neural networks for time-series prediction

The system predicts groundwater level trends using environmental and climatic parameters such as rainfall, temperature, humidity, evaporation, recharge rate, pumping index, and geographical coordinates.

The project also includes a Streamlit-based web application for interactive groundwater level prediction.

Features
Groundwater level forecasting using deep learning
Noise reduction with Wavelet Transform
Dimensionality reduction using PCA
LSTM-based sequential prediction model
Interactive Streamlit web interface
Model testing and evaluation notebooks
CSV dataset support
Tech Stack
Programming Language
Python
Libraries & Frameworks
NumPy
Pandas
Matplotlib
Scikit-learn
PyWavelets
PyTorch
Streamlit
Project Structure
Ground-water-level-forecasting-main/
│
├── genDataset.ipynb                 # Dataset preprocessing and generation
├── main.ipynb                       # Model training notebook
├── main (1).py                      # Python implementation of model
├── streamlit_app_v3.py              # Streamlit web application
├── test_model.ipynb                 # Model testing notebook
├── requirements.txt                 # Required dependencies
├── sample_test_station.csv          # Sample input dataset
├── gwl_manual_quarterly_cgwb_wb_1991_2020.csv
└── README.md
Working Methodology
1. Data Preprocessing
Cleaning and handling missing values
Feature normalization and scaling
Time-series preparation
2. DWT Denoising

Discrete Wavelet Transform is applied to remove noise from groundwater signals and improve prediction quality.

3. PCA Feature Reduction

PCA reduces high-dimensional environmental features into principal components while preserving important information.

4. LSTM Forecasting

The processed sequential data is fed into an LSTM model to forecast future groundwater levels.

Installation
Clone the Repository
git clone https://github.com/your-username/groundwater-level-forecasting.git
cd groundwater-level-forecasting
Install Dependencies
pip install -r requirements.txt
Run the Streamlit Application
streamlit run streamlit_app_v3.py
Sample Input Parameters

The model uses the following parameters:

Rainfall (mm)
Temperature (°C)
Relative Humidity (%)
Evaporation (mm)
Pumping Index
Recharge Index
Latitude
Longitude
Future Enhancements
Real-time groundwater monitoring
Integration with IoT sensors
Deployment on cloud platforms
Improved forecasting accuracy using advanced hybrid models
Visualization dashboards for groundwater analytics
Applications
Water resource management
Agricultural planning
Environmental monitoring
Smart irrigation systems
Climate impact analysis
Results

The hybrid PCA + DWT + LSTM model helps improve forecasting performance by:

Reducing noise in groundwater signals
Capturing temporal dependencies effectively
Improving prediction stability and accuracy
Author

Sanath Pendem

Artificial Intelligence & Machine Learning Enthusiast
Interested in Deep Learning and Time-Series Forecasting
License

This project is for educational and research purposes.
