# 📊 Web-Based SQL Query and Plot Recommendation App

This is a Flask web application that allows users to upload CSV files, run SQL queries, and get automated chart recommendations using a trained machine learning model.
Additional Features: Webpage supports ***light theme*** or ***dark theme*** and memory functions (remembers last modification)

## 🚀 Features

- Upload CSV files and automatically load them into an SQLite database
- Execute custom SQL queries on uploaded data
- Get chart type recommendations (bar, pie, line) based on query structure
- Visualize query results with Matplotlib
- Clean up old plots automatically
- Simple web UI with Flask & Jinja2

## 🛠️ Technologies Used

- Python 3
- Flask
- SQLite
- Pandas
- Matplotlib
- Scikit-learn + PyTorch (for ML)
- HTML/CSS (Jinja2 templates)

## 🌐 How the Webpage looks
### Section 1:
![image](https://github.com/user-attachments/assets/1921f544-1cfa-4303-ad05-4816dbb0d02c)
#### Description: Users can upload CSV files & write SQL queries to manipulate table
---
### Section 2:
![image](https://github.com/user-attachments/assets/f02ee696-29de-4486-883d-854c8b29d2c7)
#### Description: Users can choose to display, delete select, delete all, download or save previously uploaded CSV Files.
---
### Section 3:
![image](https://github.com/user-attachments/assets/42b3f9e3-57d2-471e-b5e7-3b0a2a4bedbf)
#### Description: Users can choose which data to plot by getting a recommanded plot type.
---
### Section 4:
![image](https://github.com/user-attachments/assets/e30d820a-5f2b-4b16-9373-af0924a4078b)
#### Description: Users can genereate their own 2 column table, enter table values and download. 
---


