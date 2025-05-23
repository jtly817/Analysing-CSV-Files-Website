from flask import Blueprint, render_template, request, session, send_file, current_app
from services.csv_service import CSVService
from services.sql_service import SQLService
from services.table_service import TableService
from services.ml_service import MLService
from services.plot_service import PlotService
import io

main_bp = Blueprint('main', __name__)

def get_services():
    conn = current_app.config['DATABASE_CONNECTION']
    return CSVService(conn), SQLService(conn), TableService(conn)

@main_bp.route('/')
def index():
    _, _, table_service = get_services()
    return render_template('index.html', csv_table='', tables=table_service.get_all_tables())

@main_bp.route('/upload', methods=['POST'])
def upload():
    csv_service, _, table_service = get_services()
    files = request.files.getlist('csv_file')
    messages = []

    for file in files:
        if file.filename.endswith('.csv'):
            filename, table_name = csv_service.save_csv(file)
            messages.append(f"<p class='t' style='color:green;'>Uploaded: <b>{filename}</b> → Table: <b>{table_name}</b></p>")
        else:
            messages.append(f"<p class='t' style='color:red;'>Skipped: {file.filename}</p>")

    return render_template('index.html', csv_table="\n".join(messages), tables=table_service.get_all_tables())

@main_bp.route('/SQL', methods=['POST'])
def run_sql():
    _, sql_service, table_service = get_services()
    sql_string = request.form.get('sql_query', '').strip()
    try:
        html = sql_service.run_query(sql_string)
    except Exception as e:
        html = f"<p class='t' style='color:red;'>SQL Error: {e}</p>"
    return render_template('index.html', csv_table=html, tables=table_service.get_all_tables())

@main_bp.route('/show_table', methods=['POST'])
def show_table():
    _, sql_service, table_service = get_services()
    table_name = request.form.get('table_name', '').strip()

    try:
        html = sql_service.get_table_data(table_name)
    except Exception as e:
        html = f"<p class='t' style='color:red;'>Error: {e}</p>"

    return render_template('index.html', csv_table=html, tables=table_service.get_all_tables())

@main_bp.route('/delete_selected_csv', methods=['POST'])
def delete_selected_csv():
    csv_service, _, table_service = get_services()
    table_name = request.form.get('table_name', '').strip()

    if not table_name:
        msg = "<p class='t' style='color:red;'>No table selected to delete.</p>"
    else:
        success, message = csv_service.delete_single_csv(table_name)
        msg = f"<p class='t' style='color:{'green' if success else 'red'};'>{message}</p>"

    return render_template('index.html', csv_table=msg, tables=table_service.get_all_tables())

@main_bp.route('/delete_all_csv', methods=['POST'])
def delete_all_csv():
    csv_service, _, table_service = get_services()
    deleted, dropped = csv_service.delete_all_csvs()
    session.pop('last_display', None)

    msg = ""
    msg += "".join(f"<p class='t' style='color:green;'>Deleted file: {f}</p>" for f in deleted)
    msg += "".join(f"<p class='t' style='color:orange;'>Dropped table: {t}</p>" for t in dropped)
    
    if not deleted and not dropped:
        msg = "<p class='t' style='color:blue;'>No CSV files or tables to delete.</p>"

    return render_template('index.html', csv_table=msg, tables=table_service.get_all_tables())

@main_bp.route('/download_displayed', methods=['POST'])
def download_displayed():
    _, sql_service, table_service = get_services()
    down_table_name = request.form.get("downloadTable", "").strip()

    if 'last_display' not in session:
        return render_template('index.html',
                               csv_table="<p class='t' style='color:red;'>Nothing to download.</p>",
                               tables=table_service.get_all_tables())
    if not down_table_name:
        return render_template('index.html',
                               csv_table="<p class='t' style='color:red;'>No name provided to download.</p>",
                               tables=table_service.get_all_tables())
    try:
	
        csv_data = sql_service.download_last_display()
        return send_file(io.BytesIO(csv_data.encode()), mimetype='text/csv',
                         as_attachment=True, download_name=f"{down_table_name}.csv")
    except Exception as e:
        return render_template('index.html',
                               csv_table=f"<p class='t' style='color:red;'>Download failed: {e}</p>",
                               tables=table_service.get_all_tables())

@main_bp.route('/save_displayed', methods=['POST'])
def save_displayed():
    _, _, table_service = get_services()
    new_table = request.form.get("saveTable", "").strip()

    success, msg_text = table_service.save_displayed_table(new_table)

    # Format result message
    if success:
        msg = f"<p class='t' style='color:green;'>{msg_text}</p>"
    else:
        msg = f"<p class='t' style='color:red;'>{msg_text}</p>"

    return render_template('index.html', csv_table=msg, tables=table_service.get_all_tables())

@main_bp.route('/recommend_plot', methods=['POST'])
def recommend_plot():
    csv_service, sql_service, table_service = get_services()
    ml_service = MLService()
    table_name = request.form.get("table_name")

    if not table_name:
        return render_template("index.html",
                               msg="<p class='t'>No table selected.</p>",
                               tables=table_service.get_all_tables())

    try:
        df = sql_service.get_table_dataframe(table_name)
        recommendation = ml_service.predict_plot_type(df)

        return render_template("index.html",
                               recommendation=recommendation,
                               table_name=table_name,
                               tables=table_service.get_all_tables())
    except Exception as e:
        return render_template("index.html",
                               msg=f"<p class='t' style='color:red;'>Error: {e}</p>",
                               tables=table_service.get_all_tables())

from services.plot_service import PlotService

import os
from services.plot_service import PlotService
from flask import session

@main_bp.route('/plot_data', methods=['POST'])
def plot_data():
    _, sql_service, table_service = get_services()
    plot_service = PlotService()
    
    table_name = request.form.get("table_name")
    plot_type = request.form.get("plot_type")

    if not table_name or not plot_type:
        return render_template("index.html", msg="<p class='t' style='color:red;'>Missing table or plot type.</p>", tables=table_service.get_all_tables())

    try:
        # Delete previous plot if it exists
        old_plot = session.pop('last_plot', None)
        plot_service.remove_old_plot(old_plot)

        # Get data
        df = sql_service.get_table_dataframe(table_name)

        # Generate new plot
        if plot_type.lower() == "bar":
            img_file = plot_service.plot_bar(df)
        elif plot_type.lower() == "line":
            img_file = plot_service.plot_line(df)
        elif plot_type.lower() == "pie":
            img_file = plot_service.plot_pie(df)
        else:
            raise Exception(f"Unsupported plot type: {plot_type}")

        # Save new plot filename in session
        session['last_plot'] = img_file

        return render_template("index.html", plot_url=f"/static/{img_file}", tables=table_service.get_all_tables())
    except Exception as e:
        return render_template("index.html", msg=f"<p class='t' style='color:red;'>Plot failed: {e}</p>", tables=table_service.get_all_tables())
