# 🎯 Customer Churn Prediction App
[Live Demo -](https://customerchurnpred-ann.streamlit.app/)
A fast and intuitive Streamlit web application that predicts customer churn using a neural network model. Built for banking/telecom industries to identify customers at risk of leaving.

## 🚀 Features

- **Real-time Predictions** - Get instant churn probability scores
- **Clean Interface** - Simple form-based UI for quick predictions
- **Model Transparency** - Detailed model architecture and performance metrics in sidebar
- **Production Ready** - Optimized for deployment with error handling
- **Lightweight** - Fast loading with only 11.5KB model size

## 📊 Model Performance

- **Architecture**: 3-layer Sequential Neural Network (64→32→1)
- **Training Accuracy**: 86.5%
- **Validation Accuracy**: 85.5% 
- **Model Size**: 2,945 parameters (11.5 KB)
- **Training Speed**: 6ms/step

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <your-repo-url>
cd customer-churn-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the model files in the project directory:
- `model.h5` - Trained neural network model
- `label_encoder_gender.pkl` - Gender label encoder
- `onehot_encoder_geo.pkl` - Geography one-hot encoder  
- `scaler.pkl` - Feature scaler

## 🚀 Running the App

### Local Development
```bash
streamlit run app.py
```

## 📋 Usage

1. **Fill out the customer form** with the following details:
   - Geography (France, Germany, Spain)
   - Gender (Male, Female)
   - Age (18-92)
   - Account Balance
   - Credit Score (300-850)
   - Estimated Salary
   - Tenure (0-10 years)
   - Number of Products (1-4)
   - Has Credit Card (Yes/No)
   - Is Active Member (Yes/No)

2. **Click "Predict"** to get instant results

3. **View Results**:
   - ✅ **LOW CHURN RISK** - Customer likely to stay
   - ⚠️ **HIGH CHURN RISK** - Customer at risk of leaving



## 🧠 Model Details

### Input Features (9 total)
- **Credit Score**: Customer's credit rating (300-850)
- **Geography**: Customer's country (France/Germany/Spain) 
- **Gender**: Male or Female
- **Age**: Customer age in years
- **Tenure**: Years as customer (0-10)
- **Balance**: Account balance
- **NumOfProducts**: Number of bank products (1-4)
- **HasCrCard**: Has credit card (0/1)
- **IsActiveMember**: Active customer status (0/1)
- **EstimatedSalary**: Annual salary estimate

### Model Architecture
```
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 64)                832       
dense_1 (Dense)              (None, 32)                2,080     
dense_2 (Dense)              (None, 1)                 33        
=================================================================
Total params: 2,945
Trainable params: 2,945
Non-trainable params: 0
```

## 📊 Performance Metrics

| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | 86.5% | 85.5% |
| Loss | 0.330 | 0.360 |

**Training Details:**
- Epochs: 100 (stopped at 15 for best performance)
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Batch Size: 32
- Training Speed: 6ms/step

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Originally developed as a simple prediction app
- Enhanced with Claude AI for better UX, error handling, and deployment optimization
- Built using Streamlit, TensorFlow, and scikit-learn


---

**Made with ❤️ for better customer retention analytics**