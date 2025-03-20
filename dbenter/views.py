import pandas as pd
import io
from django.shortcuts import render, redirect
from .models import DataAnalysis, CleanedData  # Ensure CleanedData is imported
from .forms import UploadCSVForm
from .models import DataAnalysis
from project1.views import get_mysql_connection
from bs4 import BeautifulSoup
import requests
from django.contrib import messages

SESSION_CSV_DATA = "csv_data"
SESSION_UNDO_STACK = "undo_stack"
SESSION_REDO_STACK = "redo_stack"
SESSION_SCRAPED_TABLES = "scraped_tables"
SESSION_SCRAPED_URL = "scraped_url"

def display_csv(request):
    """Displays uploaded CSV or scraped tables and allows data cleaning with undo/redo support."""
    if not request.session.get("is_logged_in"):
        return redirect("login")

    user_tables = []
    selected_table = request.GET.get("table_name", "")

    if request.user.is_authenticated:
        user_tables = DataAnalysis.objects.filter(user=request.user).values_list("table_name", flat=True)

        if selected_table:
            table_data = DataAnalysis.objects.filter(user=request.user, table_name=selected_table).first()
            if table_data:
                request.session[SESSION_CSV_DATA] = table_data.data  

    if request.method == "POST":
        action = request.POST.get("action")  # For undo/redo
        method = request.POST.get("method")
        column_name = request.POST.get("column")
        old_expr = request.POST.get("old_expr")
        new_expr = request.POST.get("new_expr")
        expr = request.POST.get("expr")
        trim_side = request.POST.get("trim_side")
        add_serial = request.POST.get("add_serial")
        search_query = request.POST.get("search_query")
        filter_column = request.POST.get("filter_column")
        filter_condition = request.POST.get("filter_condition")
        filter_value = request.POST.get("filter_value")
        filter_col1 = request.POST.get("filter_col1")  # First column for operations
        filter_col2 = request.POST.get("filter_col2")  # Second column for operations
        filter_op = request.POST.get("filter_op")  # Operation (+, -, *, /)
        csv_data = request.session.get(SESSION_CSV_DATA)

        # Handle Undo
        if action == "undo":
            undo_stack = request.session.get(SESSION_UNDO_STACK, [])
            redo_stack = request.session.get(SESSION_REDO_STACK, [])

            if undo_stack:
                last_state = undo_stack.pop()  # Get last saved state
                redo_stack.append(csv_data)  # Save current state for redo
                request.session[SESSION_CSV_DATA] = last_state  # Restore last state
                request.session[SESSION_UNDO_STACK] = undo_stack
                request.session[SESSION_REDO_STACK] = redo_stack

            return redirect("display_csv")

        # Handle Redo
        elif action == "redo":
            redo_stack = request.session.get(SESSION_REDO_STACK, [])
            undo_stack = request.session.get(SESSION_UNDO_STACK, [])

            if redo_stack:
                next_state = redo_stack.pop()  # Get last undone state
                undo_stack.append(csv_data)  # Save current state for undo
                request.session[SESSION_CSV_DATA] = next_state  # Restore undone state
                request.session[SESSION_UNDO_STACK] = undo_stack
                request.session[SESSION_REDO_STACK] = redo_stack

            return redirect("display_csv")

        if csv_data:
            df = pd.read_csv(io.StringIO(csv_data))

            # Save current state before applying changes
            undo_stack = request.session.get(SESSION_UNDO_STACK, [])
            undo_stack.append(df.to_csv(index=False))
            request.session[SESSION_UNDO_STACK] = undo_stack

        if action == "add_column":
            new_column_name = request.POST.get("new_column_name")
            column_1 = request.POST.get("column_1")
            column_2 = request.POST.get("column_2")
            operation = request.POST.get("operation")

            if new_column_name and column_1 and column_2 and operation:
                try:
                    if operation == "add":
                        df[new_column_name] = df[column_1] + df[column_2]
                    elif operation == "subtract":
                        df[new_column_name] = df[column_1] - df[column_2]
                    elif operation == "multiply":
                        df[new_column_name] = df[column_1] * df[column_2]
                    elif operation == "divide":
                        df[new_column_name] = df[column_1] / df[column_2]

                    # Save the new data and clear redo stack
                    request.session[SESSION_CSV_DATA] = df.to_csv(index=False)
                    request.session[SESSION_REDO_STACK] = []

                    return redirect("display_csv")

                except Exception as e:
                    messages.error(request, f"Error: {e}")

        # Handle Data Cleaning
        if csv_data:
            df = pd.read_csv(io.StringIO(csv_data))

            # Save current state before applying changes
            undo_stack = request.session.get(SESSION_UNDO_STACK, [])
            undo_stack.append(df.to_csv(index=False))
            request.session[SESSION_UNDO_STACK] = undo_stack

            if method:
                df = apply_cleaning_method(df, method, column_name, old_expr=old_expr, new_expr=new_expr, expr=expr, trim_side=trim_side)

            if add_serial:
                df["Serial Number"] = range(1, len(df) + 1)
                cols = ["Serial Number"] + [col for col in df.columns if col != "Serial Number"]
                df = df[cols]  # Reordering columns to move "Serial Number" to the front

            # Handle Search
            if search_query and column_name in df.columns:
                df = df[df[column_name].astype(str).str.contains(search_query, case=False, na=False)]









            # Handle Filter
            if request.method == "POST":
                action = request.POST.get("action")

            if action == "filter":
                 filter_column = request.POST.get("filter_column")
                 filter_value = request.POST.get("filter_value")

            if filter_col1 and filter_col2 and filter_op:
               if filter_col1 in df.columns and filter_col2 in df.columns:
                try:
                    if filter_op == "add":
                        df["temp_filter_col"] = df[filter_col1] + df[filter_col2]
                    elif filter_op == "subtract":
                        df["temp_filter_col"] = df[filter_col1] - df[filter_col2]
                    elif filter_op == "multiply":
                        df["temp_filter_col"] = df[filter_col1] * df[filter_col2]
                    elif filter_op == "divide":
                        df["temp_filter_col"] = df[filter_col1] / df[filter_col2]

                    # Apply filtering on the temporary column
                    if filter_condition and filter_value:
                        filter_value = float(filter_value) if filter_value.replace('.', '', 1).isdigit() else filter_value

                        if filter_condition == "=":
                            df = df[df["temp_filter_col"] == filter_value]
                        elif filter_condition == ">":
                            df = df[df["temp_filter_col"] > filter_value]
                        elif filter_condition == "<":
                            df = df[df["temp_filter_col"] < filter_value]
                        elif filter_condition == ">=":
                            df = df[df["temp_filter_col"] >= filter_value]
                        elif filter_condition == "<=":
                            df = df[df["temp_filter_col"] <= filter_value]

                    df.drop(columns=["temp_filter_col"], inplace=True)  # Remove temporary column

                except Exception as e:
                    messages.error(request, f"Error in column operation: {e}")

        # Handle numeric filtering
        if filter_column and filter_value and filter_column in df.columns:
            try:
                filter_value = float(filter_value) if filter_value.replace('.', '', 1).isdigit() else filter_value

                if filter_condition == "=":
                    df = df[df[filter_column] == filter_value]
                elif filter_condition == ">":
                    df = df[df[filter_column] > filter_value]
                elif filter_condition == "<":
                    df = df[df[filter_column] < filter_value]
                elif filter_condition == ">=":
                    df = df[df[filter_column] >= filter_value]
                elif filter_condition == "<=":
                    df = df[df[filter_column] <= filter_value]

            except Exception as e:
                messages.error(request, f"Error in filtering: {e}")




        # Handle text filtering
        elif action == "text_filter":
            column_name = request.POST.get("column_name")
            search_query = request.POST.get("search_query")

            if column_name and search_query and column_name in df.columns:
                try:
                    df = df[df[column_name].astype(str).str.contains(search_query, case=False, na=False)]
                except Exception as e:
                    messages.error(request, f"Error in text filtering: {e}")












        # Save new CSV data and clear redo stack
        request.session[SESSION_CSV_DATA] = df.to_csv(index=False)
        request.session[SESSION_REDO_STACK] = []

        return redirect("display_csv")


    # Retrieve and display the CSV data
    csv_data = request.session.get(SESSION_CSV_DATA)
    if csv_data:
        df = pd.read_csv(io.StringIO(csv_data))
        data_entries = df.to_dict(orient="records")
        columns = df.columns
    else:
        data_entries = []
        columns = []

    return render(request, "display_csv.html", {
        "data_entries": data_entries,
        "columns": columns,
        "user_tables": user_tables,
        "selected_table": selected_table,
    })









from django.http import JsonResponse
import pandas as pd
import io

def get_unique_values(request):
    """Fetch unique values from a column in the uploaded CSV."""
    column = request.GET.get("column", "")
    csv_data = request.session.get(SESSION_CSV_DATA)

    if csv_data and column:
        df = pd.read_csv(io.StringIO(csv_data))
        if column in df.columns:
            unique_values = df[column].dropna().unique().tolist()
            return JsonResponse({"unique_values": unique_values})

    return JsonResponse({"unique_values": []})







def csv_to_dataframe(csv_data):
    df = pd.read_csv(io.StringIO(csv_data))
    return df

def add_serial_number_column(df):
    df.insert(0, 'Serial No', range(1, len(df) + 1))
    return df


import json
from django.core.exceptions import ValidationError

import pandas as pd
from django.core.exceptions import ValidationError
from .models import CleanedData  # Ensure this model is correctly defined

def apply_cleaning_method(df, method, column_name=None, **kwargs):
    if method == 'DropCols':
        df.drop(columns=column_name, inplace=True)

    elif method == 'SetIndex':
        df.set_index(column_name, inplace=True)

    elif method == 'DropDuplicates':
        df.drop_duplicates(inplace=True)

    elif method == 'Replace':
        old_value = kwargs.get('old_expr')
        new_value = kwargs.get('new_expr')

        if df[column_name].dtype == 'object':  # String column
            df[column_name].replace(old_value, new_value, inplace=True, regex=True)
        else:  # Numeric column (int or float)
            try:
                old_value = float(old_value) if '.' in str(old_value) else int(old_value)
                new_value = float(new_value) if '.' in str(new_value) else int(new_value)
            except ValueError:
                pass  # If conversion fails, keep them as strings

            df[column_name].replace(old_value, new_value, inplace=True)


    elif method == 'DropRowIf':
        expr = kwargs.get('expr')
        condition = kwargs.get('condition', '==')  # Default condition is '=='

    # Try converting expr to numeric if column is numeric
        if df[column_name].dtype in ['int64', 'float64']:
            try:
                expr = float(expr) if '.' in str(expr) else int(expr)
            except ValueError:
                pass  # If conversion fails, keep as string

        # Ensure the column is treated as numeric
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

    # Apply filtering based on condition
        if condition == '==':
            df.drop(df[df[column_name] == expr].index, inplace=True)
        elif condition == '>':
            df.drop(df[df[column_name] > expr].index, inplace=True)  # Ensure column is numeric
        elif condition == '<':
            df.drop(df[df[column_name] < expr].index, inplace=True)
        elif condition == '>=':
            df.drop(df[df[column_name] >= expr].index, inplace=True)
        elif condition == '<=':
            df.drop(df[df[column_name] <= expr].index, inplace=True)
        elif condition == '!=':
            df.drop(df[df[column_name] != expr].index, inplace=True)
        elif condition == 'contains':  # Works for string columns
            df.drop(df[df[column_name].astype(str).str.contains(str(expr), na=False)].index, inplace=True)
        elif condition == 'startswith':  # String startswith
            df.drop(df[df[column_name].astype(str).str.startswith(str(expr), na=False)].index, inplace=True)
        elif condition == 'endswith':  # String endswith
            df.drop(df[df[column_name].astype(str).str.endswith(str(expr), na=False)].index, inplace=True)



    elif method == 'ResetIndex':
        df.reset_index(drop=True, inplace=True)

    elif method == 'Trim':
        trim_side = kwargs.get('trim_side', 'Both')
        expr = kwargs.get('expr', '')
        if df[column_name].dtype != 'object':
            df[column_name] = df[column_name].astype(str)
        if trim_side == 'Left':
            df[column_name] = df[column_name].str.lstrip(expr)
        elif trim_side == 'Right':
            df[column_name] = df[column_name].str.rstrip(expr)
        else:
            df[column_name] = df[column_name].str.strip(expr)

    elif method == 'ChangeTypeCol':
        new_type = kwargs.get('new_type', 'str')  # Get selected type from frontend

        try:
            if new_type == 'str':
                df[column_name] = df[column_name].astype(str)
            elif new_type == 'int':
                df[column_name] = df[column_name].astype(int)
            elif new_type == 'float':
                df[column_name] = df[column_name].astype(float)
            elif new_type == 'bool':
                df[column_name] = df[column_name].astype(bool)
            elif new_type == 'complex':
                df[column_name] = df[column_name].astype(complex)
            elif new_type == 'category':
                df[column_name] = df[column_name].astype('category')
            elif new_type == 'datetime':
                df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
            elif new_type == 'object':
                df[column_name] = df[column_name].astype(object)
        except ValueError:
            return df  # Return unchanged if conversion fails


    elif method in ['SortAscending', 'SortDescending']:
        df[column_name] = pd.to_numeric(df[column_name], errors='ignore')
        descending = (method == 'SortDescending')
        df.sort_values(by=column_name, ascending=not descending, inplace=True)

    elif method == 'AddColumn':
        col1 = kwargs.get('col1')
        col2 = kwargs.get('col2')
        operation = kwargs.get('operation')
        new_col = kwargs.get('new_col')

        if col1 in df.columns and col2 in df.columns:
            try:
                if operation == '+':
                    df[new_col] = df[col1] + df[col2]
                elif operation == '-':
                    df[new_col] = df[col1] - df[col2]
                elif operation == '*':
                    df[new_col] = df[col1] * df[col2]
                elif operation == '/':
                    df[new_col] = df[col1] / df[col2]
                df.fillna('', inplace=True)
            except Exception as e:
                raise ValidationError(f"Error performing operation: {e}")

    elif method == 'DropNullRows':
        df.dropna(inplace=True)

    elif method == 'Fillna':
        fill_method = kwargs.get('fill_method', 'value')
        fill_value = kwargs.get('fill_value', '')

        if fill_method == 'value':
            df.fillna(fill_value, inplace=True)

        elif fill_method == 'ffill':  # Forward Fill
            df.fillna(method='ffill', inplace=True)

        elif fill_method == 'bfill':  # Backward Fill
            df.fillna(method='bfill', inplace=True)

        elif fill_method == 'mean':  # Only numeric columns
            df.fillna(df.mean(numeric_only=True), inplace=True)

        elif fill_method == 'median':
            df.fillna(df.median(numeric_only=True), inplace=True)

        elif fill_method == 'mode':
            for col in df.columns:
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else ''
                df[col].fillna(mode_value, inplace=True)

    df.fillna('', inplace=True)  # Ensure no NaNs remain

    cleaned_data = df.to_dict(orient='records')
    CleanedData.objects.all().delete()
    CleanedData.objects.create(cleaned_data=cleaned_data)

    return df





















def upload_csv(request):
    if request.method == "POST":
        form = UploadCSVForm(request.POST, request.FILES)

        if form.is_valid():
            DataAnalysis.objects.all().delete()  # Clearing existing data
            uploaded_file = request.FILES.get('csv_file')

            if not uploaded_file:
                form.add_error('csv_file', "No file selected.")
                return render(request, 'index100.html', {'form': form})

            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()

                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                elif file_extension in ['xls', 'xlsx']:
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                else:
                    form.add_error('csv_file', "Unsupported file format. Please upload a CSV or Excel file.")
                    return render(request, 'index100.html', {'form': form})

                if df.empty:
                    form.add_error('csv_file', "The uploaded file is empty.")
                    return render(request, 'index100.html', {'form': form})

                # ✅ Convert DataFrame to CSV format and store in session
                request.session[SESSION_CSV_DATA] = df.to_csv(index=False)

                # ✅ Reset undo/redo stacks
                request.session[SESSION_UNDO_STACK] = []
                request.session[SESSION_REDO_STACK] = []

                return redirect('display_csv')  # Redirect to CSV display page

            except pd.errors.ParserError:
                form.add_error('csv_file', "Error reading the file. Ensure it is a valid CSV/Excel file.")
            except Exception as e:
                form.add_error('csv_file', f"Unexpected error: {str(e)}")

    else:
        form = UploadCSVForm()

    return render(request, 'index100.html', {'form': form})














































import requests
import pandas as pd
import io
import json
from bs4 import BeautifulSoup
from django.shortcuts import render, redirect
from .models import DataAnalysis  # Ensure you have this model

SESSION_CSV_DATA = "csv_data"
SESSION_UNDO_STACK = "undo_stack"
SESSION_REDO_STACK = "redo_stack"
SESSION_SCRAPED_TABLES = "scraped_tables"
SESSION_SCRAPED_URL = "scraped_url"


def scrape_tables(request):
    """Scrapes tables from a user-provided URL and stores them in the session."""
    if request.method == "POST":
        url = request.POST.get("url")
        request.session[SESSION_SCRAPED_TABLES] = []  # Reset previous session data
        request.session[SESSION_SCRAPED_URL] = url

        if url:
            try:
                response = requests.get(url)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")
                tables = soup.find_all("table")

                print(f"🔍 Found {len(tables)} tables on the page.")  # Debugging

                scraped_tables = []
                for index, table in enumerate(tables):
                    try:
                        df = pd.read_html(io.StringIO(str(table)))[0]  # ✅ Fix applied
                        csv_data = df.to_csv(index=False)
                        scraped_tables.append(csv_data)
                        print(f"✅ Table {index} scraped successfully!")  # Debugging
                    except Exception as e:
                        print(f"⚠️ Error processing table {index}: {str(e)}")  # Debugging

                request.session[SESSION_SCRAPED_TABLES] = scraped_tables
                print("📌 Session stored tables:", len(scraped_tables))  # Debugging

                return redirect("select_scraped_table")

            except Exception as e:
                return render(request, "index100.html", {"error": f"Error scraping data: {str(e)}"})

    return render(request, "index100.html")


def select_scraped_table(request):
    """Displays scraped tables for user selection."""
    scraped_tables = request.session.get(SESSION_SCRAPED_TABLES, [])
    print("🖥️ Tables sent to template:", len(scraped_tables))  # Debugging

    return render(request, "index100.html", {"scraped_tables": scraped_tables})


def display_scraped_table(request):
    """Displays the selected scraped table and saves it to the database."""
    table_index = request.GET.get("table_index")
    scraped_tables = request.session.get(SESSION_SCRAPED_TABLES, [])
    url = request.session.get(SESSION_SCRAPED_URL, "")

    if table_index is not None and table_index.isdigit():
        table_index = int(table_index)
        if 0 <= table_index < len(scraped_tables):
            df = pd.read_csv(io.StringIO(scraped_tables[table_index]))

            if request.user.is_authenticated:
                DataAnalysis.objects.create(
                    user=request.user,
                    table_name=f"Scraped Table {table_index}",
                    data=json.loads(df.to_json(orient="records")),
                    source_url=url,
                )

            request.session[SESSION_CSV_DATA] = df.to_csv(index=False)
            return redirect("display_csv")

    return redirect("scrape_tables")

















