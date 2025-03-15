import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QGridLayout, QLineEdit, QComboBox, 
    QPushButton, QMessageBox, QLabel, QGroupBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class PricePredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mercedes Price Prediction")
        self.resize(720, 540)  # Wider and taller for a more spacious layout
        
        # Apply a modern dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #2E2E2E;
                color: #FFFFFF;
                font-size: 14px;
            }
            QLineEdit, QComboBox {
                background-color: #3C3F41;
                color: #FFFFFF;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px;
                margin: 4px 0;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #999999;
            }
            QPushButton {
                background-color: #4C4F51;
                color: #FFFFFF;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 8px 16px;
                margin-top: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5C5F61;
            }
            QPushButton:pressed {
                background-color: #6C6F71;
            }
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 4px;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                margin: 4px 0;
                font-weight: bold;
            }
        """)

        # Main layout with margins and spacing
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Title label
        title_label = QLabel("Mercedes Price Prediction")
        title_font = QFont("Arial", 22, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Group box for input fields
        input_group = QGroupBox("Enter Car Details")
        input_layout = QGridLayout()
        input_layout.setSpacing(10)
        input_group.setLayout(input_layout)
        
        self.inputs = {}
        
        # Numeric fields
        numeric_fields = ["year", "mileage", "tax", "mpg", "engineSize"]
        row = 0
        for field in numeric_fields:
            label = QLabel(field.capitalize())
            line_edit = QLineEdit()
            self.inputs[field] = line_edit
            
            input_layout.addWidget(label, row, 0)
            input_layout.addWidget(line_edit, row, 1)
            row += 1
        
        # Model ComboBox
        label_model = QLabel("Model")
        self.model_combo = QComboBox()
        model_options = [
            "A Class", "B Class", "C Class", "CL Class", "CLA Class", "CLC Class", "CLK",
            "CLS Class", "E Class", "G Class", "GL Class", "GLA Class", "GLB Class", "GLC Class",
            "GLE Class", "GLS Class", "M Class", "R Class", "S Class", "SL CLASS", "SLK",
            "V Class", "X-CLASS", "180", "200", "220", "230"
        ]
        self.model_combo.addItems(model_options)
        self.inputs["model"] = self.model_combo
        input_layout.addWidget(label_model, row, 0)
        input_layout.addWidget(self.model_combo, row, 1)
        row += 1
        
        # Transmission ComboBox
        label_trans = QLabel("Transmission")
        self.trans_combo = QComboBox()
        trans_options = ["Automatic", "Manual", "Other", "Semi-Auto"]
        self.trans_combo.addItems(trans_options)
        self.inputs["transmission"] = self.trans_combo
        input_layout.addWidget(label_trans, row, 0)
        input_layout.addWidget(self.trans_combo, row, 1)
        row += 1
        
        # Fuel Type ComboBox
        label_fuel = QLabel("Fuel Type")
        self.fuel_combo = QComboBox()
        fuel_options = ["Diesel", "Hybrid", "Other", "Petrol"]
        self.fuel_combo.addItems(fuel_options)
        self.inputs["fuelType"] = self.fuel_combo
        input_layout.addWidget(label_fuel, row, 0)
        input_layout.addWidget(self.fuel_combo, row, 1)
        row += 1
        
        # Add the group box to the main layout
        main_layout.addWidget(input_group)
        
        # Predict button (centered)
        self.predict_button = QPushButton("Predict Price")
        self.predict_button.clicked.connect(self.predict_price)
        main_layout.addWidget(self.predict_button, alignment=Qt.AlignCenter)
        
        # Load the saved model and scaler
        try:
            self.model = tf.keras.models.load_model("my_model.h5", compile=False)
            self.scaler = joblib.load("scaler.pkl")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading model or scaler: {e}")
            sys.exit(1)
        
        # Expected columns from training (40 columns, minus 'price')
        self.expected_columns = [
            'year', 'mileage', 'tax', 'mpg', 'engineSize', 'model_ A Class',
            'model_ B Class', 'model_ C Class', 'model_ CL Class', 'model_ CLA Class',
            'model_ CLC Class', 'model_ CLK', 'model_ CLS Class', 'model_ E Class',
            'model_ G Class', 'model_ GL Class', 'model_ GLA Class', 'model_ GLB Class',
            'model_ GLC Class', 'model_ GLE Class', 'model_ GLS Class', 'model_ M Class',
            'model_ R Class', 'model_ S Class', 'model_ SL CLASS', 'model_ SLK',
            'model_ V Class', 'model_ X-CLASS', 'model_180', 'model_200', 'model_220',
            'model_230', 'transmission_Automatic', 'transmission_Manual',
            'transmission_Other', 'transmission_Semi-Auto', 'fuelType_Diesel',
            'fuelType_Hybrid', 'fuelType_Other', 'fuelType_Petrol'
        ]
    
    def predict_price(self):
        try:
            # Prepare a dictionary with zeros for each expected feature
            input_dict = {col: 0 for col in self.expected_columns}
            
            # Fill numeric features from QLineEdit inputs
            input_dict["year"] = float(self.inputs["year"].text())
            input_dict["mileage"] = float(self.inputs["mileage"].text())
            input_dict["tax"] = float(self.inputs["tax"].text())
            input_dict["mpg"] = float(self.inputs["mpg"].text())
            input_dict["engineSize"] = float(self.inputs["engineSize"].text())
            
            # One-hot encode categorical features
            chosen_model = self.inputs["model"].currentText().strip()
            model_col = f"model_ {chosen_model}"
            if model_col in input_dict:
                input_dict[model_col] = 1
            
            chosen_trans = self.inputs["transmission"].currentText().strip()
            trans_col = f"transmission_{chosen_trans}"
            if trans_col in input_dict:
                input_dict[trans_col] = 1
            
            chosen_fuel = self.inputs["fuelType"].currentText().strip()
            fuel_col = f"fuelType_{chosen_fuel}"
            if fuel_col in input_dict:
                input_dict[fuel_col] = 1
            
            # Create a single-row DataFrame in the same column order
            input_df = pd.DataFrame([input_dict], columns=self.expected_columns)
            
            # Scale the features
            X_scaled = self.scaler.transform(input_df.values)
            
            # Predict with the loaded model
            prediction = self.model.predict(X_scaled)
            predicted_price = prediction[0][0]
            
            # Show the result
            QMessageBox.information(
                self, 
                "Predicted Price", 
                f"The predicted price is: ${predicted_price:,.2f}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PricePredictor()
    window.show()
    sys.exit(app.exec_())
