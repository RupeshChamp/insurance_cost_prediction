import sys
import joblib
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit,
    QComboBox, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QGroupBox
)
from PyQt5.QtCore import Qt


class InsuranceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Insurance Cost Predictor")
        self.setFixedSize(450, 550)

        # Load the trained model
        try:
            self.model = joblib.load("medical_insurance_best_model.pkl")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Model file not found! Please save your model first.\n{e}")
            sys.exit()

        self.init_ui()

    def init_ui(self):
        # Main widget & layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # Title Label
        title_label = QLabel("Medical Insurance Prediction")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title_label)

        # Form Group Box
        form_group = QGroupBox("Enter Patient Details")
        form_layout = QVBoxLayout()

        # Age
        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("e.g. 30")
        form_layout.addWidget(QLabel("Age:"))
        form_layout.addWidget(self.age_input)

        # Sex
        self.sex_combo = QComboBox()
        self.sex_combo.addItems(["female", "male"])
        form_layout.addWidget(QLabel("Sex:"))
        form_layout.addWidget(self.sex_combo)

        # BMI
        self.bmi_input = QLineEdit()
        self.bmi_input.setPlaceholderText("e.g. 28.5")
        form_layout.addWidget(QLabel("BMI (Body Mass Index):"))
        form_layout.addWidget(self.bmi_input)

        # Children
        self.children_combo = QComboBox()
        self.children_combo.addItems(["0", "1", "2", "3", "4", "5"])
        form_layout.addWidget(QLabel("Number of Children:"))
        form_layout.addWidget(self.children_combo)

        # Smoker
        self.smoker_combo = QComboBox()
        self.smoker_combo.addItems(["no", "yes"])
        form_layout.addWidget(QLabel("Smoker Status:"))
        form_layout.addWidget(self.smoker_combo)

        # Region
        self.region_combo = QComboBox()
        self.region_combo.addItems(["northeast", "northwest", "southeast", "southwest"])
        form_layout.addWidget(QLabel("Region:"))
        form_layout.addWidget(self.region_combo)

        form_group.setLayout(form_layout)
        layout.addWidget(form_group)

        # Predict Button
        self.predict_btn = QPushButton("Predict Insurance Cost")
        self.predict_btn.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold; font-size: 14px; padding: 10px;")
        self.predict_btn.clicked.connect(self.make_prediction)
        layout.addWidget(self.predict_btn)

        # Result Display Label
        self.result_label = QLabel("Predicted Cost: $0.00")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #e74c3c; margin-top: 15px;")
        layout.addWidget(self.result_label)

        main_widget.setLayout(layout)

    def make_prediction(self):
        try:
            # Validate and fetch inputs
            age = int(self.age_input.text())
            sex = self.sex_combo.currentText()
            bmi = float(self.bmi_input.text())
            children = int(self.children_combo.currentText())
            smoker = self.smoker_combo.currentText()
            region = self.region_combo.currentText()

            # Backend Processing & Encoding matching training format
            # Sex: female -> 0, male -> 1 (matching LabelEncoder)
            sex_encoded = 1 if sex == 'male' else 0

            # Smoker: no -> 0, yes -> 1 (matching LabelEncoder)
            smoker_encoded = 1 if smoker == 'yes' else 0

            # One-Hot encoding for regions
            region_northeast = 1 if region == 'northeast' else 0
            region_northwest = 1 if region == 'northwest' else 0
            region_southeast = 1 if region == 'southeast' else 0
            region_southwest = 1 if region == 'southwest' else 0

            # Create DataFrame with exact training columns order
            input_data = pd.DataFrame([[
                age, sex_encoded, bmi, children, smoker_encoded,
                region_northeast, region_northwest, region_southeast, region_southwest
            ]], columns=[
                'age', 'sex', 'bmi', 'children', 'smoker',
                'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest'
            ])

            # Predict using model
            prediction = self.model.predict(input_data)[0]

            # Show Result
            self.result_label.setText(f"Predicted Cost: ${prediction:,.2f}")

        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values for Age and BMI!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during prediction:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InsuranceApp()
    window.show()
    sys.exit(app.exec_())