# PacketShield – Network Intrusion Detection System

PacketShield is a machine learning–based Network Intrusion Detection System designed to preprocess raw network traffic, train custom ML models, visualize performance metrics, and detect attacks in real time using live packet capture.

---

##  Team Members
- **Kaushika C (23PW12)**
- **Samantha W (23PW25)**

---

##  Features
- Preprocessing of raw network packets (log transform, clipping, robust normalization)  
- Custom-built **Random Forest** and **Gaussian Naive Bayes** classifiers  
- Accuracy, precision, recall, and F1-score evaluation  
- Confusion matrices, ROC curves, and comparison visualizations  
- Model explainability (decision paths & feature contributions)  
- **Live packet capture** using PyShark + Wireshark/tshark  
- Export of trained models and classification results  

---

##  Tech Stack
- **Python**, **Streamlit**  
- **NumPy**, **Pandas**  
- **Matplotlib**, **Seaborn**  
- **PyShark**, **tshark/Wireshark**  
- **Joblib**, **scikit-learn utilities**

---

##  Project Structure
PacketShield/
│── app.py
│── requirements.txt
│── packages.txt
│── converted_csvs/ # Input datasets
│── comparison_outputs/ # Model files & results
└── README.md


---

##  How It Works
1. Place CSV datasets inside the `converted_csvs/` folder.  
2. Configure model parameters from the Streamlit sidebar.  
3. PacketShield preprocesses traffic data and extracts selected features.  
4. Custom ML models (RF & NB) are trained and evaluated.  
5. The system visualizes metrics like confusion matrices and ROC curves.  
6. Live packet capture uses **tshark** to classify traffic in real time.  
7. Outputs, including predictions and model files, are stored in `comparison_outputs/`.

##  Conclusion

PacketShield demonstrates a complete end-to-end pipeline for network intrusion detection, combining raw traffic preprocessing, custom machine learning algorithms, and real-time packet monitoring. By integrating visualization tools, explainability features, and live capture capabilities, the system provides both analytical insight and practical threat detection. This project showcases how lightweight, custom-built ML models can effectively support cybersecurity workflows and offers a foundation for further enhancements such as deep learning integration or advanced traffic analysis.
