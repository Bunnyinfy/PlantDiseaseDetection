---

# Plant Disease Detection

This project is a machine learning-based web application designed to predict plant diseases from leaf images. The application uses a pre-trained Convolutional Neural Network (CNN) model and is deployed using **Streamlit** for a simple and interactive user experience.

---

## Features

- **Image Upload**: Users can upload an image of a plant leaf to check for diseases.
- **Disease Detection**: The model analyzes the uploaded image and predicts the disease, if present.
- **Confidence Scores**: Provides the probability of each disease class for transparency.
- **User-Friendly Interface**: Built with Streamlit for an intuitive and responsive UI.

---

## Tech Stack

### **Frontend**
- **Streamlit**: For creating the web interface.

### **Backend**
- **Python**: Core programming language.
- **TensorFlow/Keras**: For the CNN model.
- **NumPy & Pandas**: For data manipulation.

### **Deployment**
- **Streamlit Sharing** : Application hosting.

---

## Installation
Before following the below steps make sure you execute the Jupyter Source file and extract h5 model(obtain your own credentials , download the dataset from the link given below)

Follow these steps to set up the application locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Bunnyinfy/plant-disease-prediction.git
   cd plant-disease-prediction
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the App**:
   Open the provided localhost link in your browser.

---

## Usage

1. Upload an image of a plant leaf through the application.
2. Wait for the model to process and predict the disease.
3. View the prediction result and confidence score.

---

## Model Details

- **Architecture**: CNN trained on a dataset of plant leaf images.
- **Input**: Images resized to a standard dimension (e.g., 224x224 pixels).
- **Output**: Disease class (e.g., healthy, bacterial spot, powdery mildew).

---

## Example Outputs

- **Input Image**: 
  Upload an image of a plant leaf.

- **Predicted Output**:
  - **Disease**: Bacterial Spot
  - **Confidence Score**: 92%

---

## Contributing

Contributions are welcome! Feel free to raise issues or submit pull requests.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- Plant dataset sourced from [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).
- Frameworks and tools: TensorFlow, Streamlit.

---

