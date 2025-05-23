import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os
import uuid

class PlotService:
    def __init__(self, output_dir="static"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _generate_filename(self):
        return f"{'last_plot'}.png"
    
    def remove_old_plot(self, old_plot):
        if old_plot:
            old_path = os.path.join("webpageSQL", "static", old_plot)
            if os.path.exists(old_path):
                os.remove(old_path)

    def _save_plot(self):
        filename = self._generate_filename()
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        return filename

    def plot_bar(self, df):
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            raise Exception("No numeric columns available for bar plot.")
        df[numeric_cols].mean().plot(kind='bar', title="Bar Chart")
        return self._save_plot()

    def plot_line(self, df):
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            raise Exception("No numeric columns available for line plot.")
        df[numeric_cols].plot(title="Line Chart")
        return self._save_plot()

    def plot_pie(self, df):
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            raise Exception("No numeric columns available for pie chart.")
        df[numeric_cols[0]].value_counts().plot(kind='pie', autopct='%1.1f%%', title="Pie Chart")
        return self._save_plot()
