from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from services.models import services
from news.models import News
from dbenter.models import User
from dbenter.models import DataAnalysis

from django.core.paginator import Paginator
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from django.contrib.auth import authenticate, login as auth_login

SESSION_CSV_DATA = 'csv_data'
SESSION_CLEANED_DATA = 'cleaned_data'


from django.shortcuts import render, redirect
from dbenter.models import User
from django.contrib.auth.hashers import make_password, check_password
from django.contrib import messages


def signup(request):
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]
        password = make_password(request.POST["password"])  # Hash password

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already exists!")
            return redirect("signup")

        User.objects.create(username=username, email=email, password=password)
        messages.success(request, "Signup successful! Please log in.")
        return redirect("login")

    return render(request, "signup.html")

def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")

        try:
            user = User.objects.get(email=email) 

            if check_password(password, user.password):  
                request.session["username"] = user.username 
                request.session["is_logged_in"] = True  
                messages.success(request, f"Welcome, {user.username}!")
                return redirect("homepage")  
            else:
                messages.error(request, "Invalid credentials!")
        except User.DoesNotExist:
            messages.error(request, "User does not exist!")

    return render(request, "login.html")

def logout_view(request):
    request.session.flush()  
    messages.success(request, "You have been logged out!")
    return redirect("login")

def homepage(request):
    if not request.session.get("is_logged_in"):  
        messages.info(request, "You need to log in first.")
        return redirect("login")  

    return render(request, "index100.html", {"username": request.session.get("username")})      


import csv
from django.http import HttpResponse
from dbenter.models import CleanedData

def download_csv(request):
    cleaned_data_entry = CleanedData.objects.last()  
    
    if not cleaned_data_entry:
        return HttpResponse("No cleaned data available for download.", content_type="text/plain")

    cleaned_data = cleaned_data_entry.cleaned_data  

    if not isinstance(cleaned_data, list) or not cleaned_data:
        return HttpResponse("Cleaned data format is incorrect.", content_type="text/plain")

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="cleaned_data.csv"'

    writer = csv.writer(response)

    headers = cleaned_data[0].keys()  
    writer.writerow(headers)

    for row in cleaned_data:
        writer.writerow(row.values())

    return response







# def csv_to_dataframe(csv_data):
#     csv_file = StringIO(csv_data)
#     df = pd.read_csv(csv_file)
#     return df









from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np



def csv_to_dataframe(csv_data):
    """Convert stored session CSV data into a Pandas DataFrame."""
    from io import StringIO
    return pd.read_csv(StringIO(csv_data))

# def analysis(request):
#     if not request.session.get("is_logged_in"):
#         messages.info(request, "You need to log in first.")
#         return redirect("login")

#     if request.method == "POST":
#         csv_data = request.session.get(SESSION_CSV_DATA)
#         if not csv_data:
#             return HttpResponse("Error: CSV data missing. Please re-upload the file.")

#         df = csv_to_dataframe(csv_data)
#         print("Available columns:", df.columns.tolist())

#         selected_columns = request.POST.getlist('columns')
#         operation = request.POST.get('operation')

#         missing_columns = [col for col in selected_columns if col not in df.columns]
#         if missing_columns:
#             return HttpResponse(f"Error: Columns {missing_columns} not found in the data.")

#         result = {}
#         row_data = []

#         numeric_data = df[selected_columns].apply(pd.to_numeric, errors='coerce')
#         is_numeric = numeric_data.notna().any()

#         if operation in ['Max', 'Min', 'Average', 'Median', 'Sum', 'Standard Deviation', 'Variance']:
#             if is_numeric.any():
#                 if operation == 'Max':
#                     result = numeric_data.max().to_dict()
#                     row_indices = {col: df[df[col] == result[col]].index.tolist() for col in selected_columns}
#                 elif operation == 'Min':
#                     result = numeric_data.min().to_dict()
#                     row_indices = {col: df[df[col] == result[col]].index.tolist() for col in selected_columns}
#                 elif operation == 'Average':
#                     result = numeric_data.mean().to_dict()
#                     row_indices = {}
#                 elif operation == 'Median':
#                     result = numeric_data.median().to_dict()
#                     row_indices = {}
#                 elif operation == 'Sum':
#                     result = numeric_data.sum().to_dict()
#                     row_indices = {}
#                 elif operation == 'Standard Deviation':
#                     result = numeric_data.std().to_dict()
#                     row_indices = {}
#                 elif operation == 'Variance':
#                     result = numeric_data.var().to_dict()
#                     row_indices = {}

#                 # Fetch row data for max/min operations
#                 if operation in ['Max', 'Min']:
#                     for col, indices in row_indices.items():
#                         for index in indices:
#                             row_data.append(df.loc[index].to_dict())

#         elif operation in ['Count Unique', 'Most Frequent', 'Count Nulls', 'String Length', 'Concatenation', 'Frequency Count']:
#             for col in selected_columns:
#                 if operation == 'Count Unique':
#                     result[col] = df[col].nunique()
#                 elif operation == 'Most Frequent':
#                     result[col] = df[col].mode().iloc[0] if not df[col].mode().empty else None
#                 elif operation == 'Count Nulls':
#                     result[col] = df[col].isnull().sum()
#                 elif operation == 'String Length':
#                     result[col] = df[col].str.len().tolist() if df[col].dtype == 'object' else None
#                 elif operation == 'Concatenation':
#                     if len(selected_columns) > 1:
#                         result[col] = df[selected_columns].astype(str).agg(' '.join, axis=1).tolist()
#                 elif operation == 'Frequency Count':
#                     result[col] = df[col].value_counts().to_dict()

#         request.session[SESSION_CLEANED_DATA] = df.to_json()
#         request.session.modified = True

#         return render(request, 'analysis.html', {
#             'result': result,
#             'row_data': row_data,
#             'columns': selected_columns,
#             'username': request.session.get("username")
#         })

#     csv_data = request.session.get(SESSION_CSV_DATA)
#     columns = csv_to_dataframe(csv_data).columns.tolist() if csv_data else []

#     return render(request, 'analysis.html', {
#         'columns': columns,
#         'result': None,
#         'row_data': None,
#         'username': request.session.get("username")
#     })













# import plotly.express as px


# SESSION_CLEANED_DATA = "cleaned_csv_data"

# def generate_graph(request):
#     if not request.session.get("is_logged_in"):
#         return redirect("login")

#     if request.method == "POST":
#         selected_columns = request.POST.getlist('columns')
#         graph_type = request.POST.get('graph_type')
        
#         top_entries = request.POST.get('top_entries', None)
#         range_start = request.POST.get('range_start', None)
#         range_end = request.POST.get('range_end', None)

#         csv_data = request.session.get(SESSION_CLEANED_DATA)
#         if not csv_data:
#             return HttpResponse("No cleaned data found. Please upload and clean the data first.", status=400)

#         df = pd.read_json(io.StringIO(csv_data))

#         if not set(selected_columns).issubset(df.columns):
#             return HttpResponse("Selected columns not found in data", status=400)

#         try:
#             if top_entries:
#                 top_entries = int(top_entries)
#                 df = df.head(top_entries)
#             elif range_start and range_end:
#                 range_start = int(range_start)
#                 range_end = int(range_end)
#                 df = df.iloc[range_start:range_end]
#         except ValueError:
#             return HttpResponse("Invalid numeric input for filtering range or top entries", status=400)

#         x_column = selected_columns[0] if df[selected_columns[0]].dtype == 'object' else None
#         y_columns = [col for col in selected_columns if col != x_column]

#         try:
#             fig = None
#             if graph_type == "Bar Graph":
#                 fig = px.bar(df, x=x_column, y=y_columns) if x_column else px.bar(df[y_columns])
#             elif graph_type == "Histogram":
#                 fig = px.histogram(df, x=y_columns[0])
#             elif graph_type == "Line Chart":
#                 fig = px.line(df, x=x_column, y=y_columns) if x_column else px.line(df[y_columns])
#             elif graph_type == "Scatter Plot":
#                 if len(y_columns) == 1 and x_column:
#                     fig = px.scatter(df, x=x_column, y=y_columns[0])
#                 elif len(y_columns) == 2:
#                     fig = px.scatter(df, x=y_columns[0], y=y_columns[1])
#                 else:
#                     return HttpResponse("Scatter plot requires a categorical x-axis or exactly 2 numerical columns", status=400)
#             elif graph_type == "Pie Chart":
#                 if len(y_columns) == 1 and x_column:
#                     fig = px.pie(df, names=x_column, values=y_columns[0])
#                 else:
#                     return HttpResponse("Pie chart requires one numerical column and one categorical column", status=400)
#             elif graph_type == "Box Plot":
#                 fig = px.box(df, y=y_columns)
            
#             # New Graph Types
#             elif graph_type == "Area Chart":
#                 fig = px.area(df, x=x_column, y=y_columns) if x_column else px.area(df[y_columns])
#             elif graph_type == "Heatmap":
#                 if len(selected_columns) == 2:
#                     fig = px.density_heatmap(df, x=selected_columns[0], y=selected_columns[1])
#                 else:
#                     return HttpResponse("Heatmap requires exactly 2 selected columns", status=400)
#             elif graph_type == "Violin Plot":
#                 fig = px.violin(df, x=x_column, y=y_columns[0], box=True, points="all") if x_column else px.violin(df, y=y_columns[0])
#             elif graph_type == "Sunburst Chart":
#                 if len(selected_columns) >= 2:
#                     fig = px.sunburst(df, path=selected_columns, values=y_columns[0])
#                 else:
#                     return HttpResponse("Sunburst requires at least 2 selected categorical columns", status=400)
#             elif graph_type == "Density Contour":
#                 if len(selected_columns) == 2:
#                     fig = px.density_contour(df, x=selected_columns[0], y=selected_columns[1])
#                 else:
#                     return HttpResponse("Density contour requires exactly 2 numerical columns", status=400)
#             elif graph_type == "Treemap":
#                 if len(selected_columns) >= 2:
#                     fig = px.treemap(df, path=selected_columns, values=y_columns[0])
#                 else:
#                     return HttpResponse("Treemap requires at least 2 selected categorical columns", status=400)
#             elif graph_type == "Funnel Chart":
#                 if len(selected_columns) >= 2:
#                     fig = px.funnel(df, x=selected_columns[0], y=selected_columns[1])
#                 else:
#                     return HttpResponse("Funnel chart requires at least 2 selected columns", status=400)
#             elif graph_type == "Radar Chart":
#                 if len(selected_columns) >= 3:
#                     fig = px.line_polar(df, r=selected_columns[1], theta=selected_columns[0], line_close=True)
#                 else:
#                     return HttpResponse("Radar chart requires at least 3 selected columns", status=400)
#             else:
#                 return HttpResponse("Invalid graph type selected", status=400)

#             graph_html = fig.to_html(full_html=False)
#             return render(request, 'graphs.html', {'graph_html': graph_html, 'columns': df.columns.tolist()})

#         except Exception as e:
#             return HttpResponse(f"Error generating graph: {str(e)}", status=500)
    
#     return HttpResponse("Invalid request method", status=400)


# def graphs_view(request):
#     if not request.session.get("is_logged_in"):
#         return redirect("login")
#     csv_data = request.session.get(SESSION_CLEANED_DATA)
#     columns = []
#     if csv_data:
#         df = pd.read_json(io.StringIO(csv_data))
#         columns = df.columns.tolist()
#     return render(request, 'graphs.html', {'columns': columns})
















import pandas as pd
import io
import plotly.express as px
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages

SESSION_CSV_DATA = "csv_data"





def analysis(request):
    if not request.session.get("is_logged_in"):
        messages.info(request, "You need to log in first.")
        return redirect("login")

    if request.method == "POST":
        csv_data = request.session.get(SESSION_CSV_DATA)
        if not csv_data:
            return HttpResponse("Error: CSV data missing. Please re-upload the file.")


        df = pd.read_csv(io.StringIO(csv_data))
        selected_columns = request.POST.getlist('columns')
        operation = request.POST.get('operation')

        try:
            min_row = int(request.POST.get('min_row', 0))  # Default to 0
            max_row = int(request.POST.get('max_row', len(df) - 1))  # Default to last index
            df = df.iloc[min_row:max_row + 1]  # Select the range
        except ValueError:
            return HttpResponse("Error: Invalid row range values.")

        # Min-Max Value Filtering (optional)
        try:
            value_min = request.POST.get('value_min')
            value_max = request.POST.get('value_max')
            if value_min and value_max:
                value_min, value_max = float(value_min), float(value_max)
                for col in selected_columns:
                    if col in df.select_dtypes(include=['number']).columns:
                        df = df[(df[col] >= value_min) & (df[col] <= value_max)]
        except ValueError:
            return HttpResponse("Error: Invalid min/max value range.")



        missing_columns = [col for col in selected_columns if col not in df.columns]
        if missing_columns:
            return HttpResponse(f"Error: Columns {missing_columns} not found in the data.")

        result = {}
        row_data = []

        numeric_data = df[selected_columns].apply(pd.to_numeric, errors='coerce')
        is_numeric = numeric_data.notna().any()

        row_indices = {}
        
        if operation in ['Max', 'Min', 'Average', 'Median', 'Sum', 'Standard Deviation', 'Variance']:
            if is_numeric.any():
                if operation == 'Max':
                    result = numeric_data.max().to_dict()
                    row_indices = {col: df[df[col] == result[col]].index.tolist() for col in selected_columns}
                elif operation == 'Min':
                    result = numeric_data.min().to_dict()
                    row_indices = {col: df[df[col] == result[col]].index.tolist() for col in selected_columns}
                elif operation == 'Average':
                    result = numeric_data.mean().to_dict()
                elif operation == 'Median':
                    result = numeric_data.median().to_dict()
                elif operation == 'Sum':
                    result = numeric_data.sum().to_dict()
                elif operation == 'Standard Deviation':
                    result = numeric_data.std().to_dict()
                elif operation == 'Variance':
                    result = numeric_data.var().to_dict()

                # Fetch row data for Max/Min operations
                if operation in ['Max', 'Min']:
                    for col, indices in row_indices.items():
                        for index in indices:
                            row_data.append(df.loc[index].to_dict())

        elif operation in ['Count Unique', 'Most Frequent', 'Count Nulls', 'String Length', 'Concatenation', 'Frequency Count']:
            for col in selected_columns:
                if operation == 'Count Unique':
                    result[col] = df[col].nunique()
                elif operation == 'Most Frequent':
                    result[col] = df[col].mode().iloc[0] if not df[col].mode().empty else None
                elif operation == 'Count Nulls':
                    result[col] = df[col].isnull().sum()
                elif operation == 'String Length':
                    result[col] = df[col].str.len().tolist() if df[col].dtype == 'object' else None
                elif operation == 'Concatenation':
                    result[col] = df[selected_columns].astype(str).agg(' '.join, axis=1).tolist()
                elif operation == 'Frequency Count':
                    result[col] = df[col].value_counts().to_dict()

        return render(request, 'analysis.html', {
            'result': result,
            'row_data': row_data,
            'columns': df.columns.tolist(),
            'username': request.session.get("username"),
            'min_row': min_row,
            'max_row': max_row,
            'value_min': request.POST.get('value_min', ''),
            'value_max': request.POST.get('value_max', '')
        })

    csv_data = request.session.get(SESSION_CSV_DATA)
    columns = pd.read_csv(io.StringIO(csv_data)).columns.tolist() if csv_data else []
    
    return render(request, 'analysis.html', {
        'columns': columns,
        'result': None,
        'row_data': None,
        'username': request.session.get("username"),
        'min_row': '',
        'max_row': '',
        'value_min': '',
        'value_max': ''
    })









#def generate_graph(request):
    if not request.session.get("is_logged_in"):
        return redirect("login")

    if request.method == "POST":
        selected_columns = request.POST.getlist('columns')
        graph_type = request.POST.get('graph_type')
        
        top_entries = request.POST.get('top_entries', None)
        range_start = request.POST.get('range_start', None)
        range_end = request.POST.get('range_end', None)
        filter_column = request.POST.get('filter_column', None)
        min_value = request.POST.get('min_value', None)
        max_value = request.POST.get('max_value', None)
        csv_data = request.session.get(SESSION_CSV_DATA)
        if not csv_data:
            return HttpResponse("No data found. Please upload and analyze the data first.", status=400)

        df = pd.read_csv(io.StringIO(csv_data))

        if not set(selected_columns).issubset(df.columns):
            return HttpResponse("Selected columns not found in data", status=400)

        try:
            if top_entries:
                top_entries = int(top_entries)
                df = df.head(top_entries)
            elif range_start and range_end:
                range_start = int(range_start)
                range_end = int(range_end)
                df = df.iloc[range_start:range_end]
            if filter_column and filter_column in df.columns:
                if min_value:
                    df = df[df[filter_column] >= float(min_value)]
                if max_value:
                    df = df[df[filter_column] <= float(max_value)]
        except ValueError:
            return HttpResponse("Invalid numeric input for filtering range or top entries", status=400)

        try:
            fig = None
            if graph_type == "Bar Graph":
                fig = px.bar(df, x=selected_columns[0], y=selected_columns[1:])
            elif graph_type == "Histogram":
                fig = px.histogram(df, x=selected_columns[0])
            elif graph_type == "Line Chart":
                fig = px.line(df, x=selected_columns[0], y=selected_columns[1:])
            elif graph_type == "Scatter Plot":
                fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1])
            elif graph_type == "Pie Chart":
                fig = px.pie(df, names=selected_columns[0], values=selected_columns[1])
            elif graph_type == "Box Plot":
                fig = px.box(df, y=selected_columns)
            elif graph_type == "Heatmap":
                fig = px.density_heatmap(df, x=selected_columns[0], y=selected_columns[1])
            elif graph_type == "Violin Plot":
                fig = px.violin(df, x=selected_columns[0], y=selected_columns[1], box=True, points="all")
            elif graph_type == "Sunburst Chart":
                fig = px.sunburst(df, path=selected_columns, values=selected_columns[1])
            elif graph_type == "Treemap":
                fig = px.treemap(df, path=selected_columns, values=selected_columns[1])
            elif graph_type == "Area Chart":
                fig = px.area(df, x=selected_columns[0], y=selected_columns[1:])
            elif graph_type == "Bubble Chart":
                fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1], size=selected_columns[1])
            elif graph_type == "Funnel Chart":
                fig = px.funnel(df, x=selected_columns[0], y=selected_columns[1])
            elif graph_type == "Density Contour":
                fig = px.density_contour(df, x=selected_columns[0], y=selected_columns[1])
            elif graph_type == "Parallel Coordinates Plot":
                fig = px.parallel_coordinates(df, dimensions=selected_columns)
            elif graph_type == "Map Plot":
                fig = px.scatter_geo(df, lat=selected_columns[0], lon=selected_columns[1])
            else:
                return HttpResponse("Invalid graph type selected", status=400)
            
            graph_html = fig.to_html(full_html=False)
            return render(request, 'graphs.html', {'graph_html': graph_html, 'columns': df.columns.tolist()})
        except Exception as e:
            return HttpResponse(f"Error generating graph: {str(e)}", status=500)
    
    return HttpResponse("Invalid request method", status=400)






def generate_graph(request):
    if not request.session.get("is_logged_in"):
        return redirect("login")

    if request.method == "POST":
        selected_columns = request.POST.getlist('columns')
        graph_type = request.POST.get('graph_type')

        top_entries = request.POST.get('top_entries', None)
        range_start = request.POST.get('range_start', None)
        range_end = request.POST.get('range_end', None)
        filter_column = request.POST.get('filter_column', None)
        min_value = request.POST.get('min_value', None)
        max_value = request.POST.get('max_value', None)

        csv_data = request.session.get(SESSION_CSV_DATA)
        if not csv_data:
            return HttpResponse("No data found. Please upload and analyze the data first.", status=400)

        df = pd.read_csv(io.StringIO(csv_data))

        if not set(selected_columns).issubset(df.columns):
            return HttpResponse("Selected columns not found in data", status=400)

        try:
            if top_entries:
                top_entries = int(top_entries)
                df = df.head(top_entries)
            elif range_start and range_end:
                range_start = int(range_start)
                range_end = int(range_end)
                df = df.iloc[range_start:range_end]

            if filter_column and filter_column in df.columns:
                if min_value:
                    df = df[df[filter_column] >= float(min_value)]
                if max_value:
                    df = df[df[filter_column] <= float(max_value)]
        except ValueError:
            return HttpResponse("Invalid numeric input for filtering range or top entries", status=400)

        try:
            fig = None
            if graph_type == "Bar Graph":
                try:
                    fig = px.bar(df, x=selected_columns[0], y=selected_columns[1:])
                except Exception as e:
                    return HttpResponse(f"Error creating Bar Graph: {str(e)}", status=500)

            elif graph_type == "Histogram":
                try:
                    fig = px.histogram(df, x=selected_columns[0])
                except Exception as e:
                    return HttpResponse(f"Error creating Histogram: {str(e)}", status=500)

            elif graph_type == "Line Chart":
                try:
                    fig = px.line(df, x=selected_columns[0], y=selected_columns[1:])
                except Exception as e:
                    return HttpResponse(f"Error creating Line Chart: {str(e)}", status=500)

            elif graph_type == "Scatter Plot":
                try:
                    fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1])
                except Exception as e:
                    return HttpResponse(f"Error creating Scatter Plot: {str(e)}", status=500)

            elif graph_type == "Pie Chart":
                try:
                    fig = px.pie(df, names=selected_columns[0], values=selected_columns[1])
                except Exception as e:
                    return HttpResponse(f"Error creating Pie Chart: {str(e)}", status=500)

            elif graph_type == "Box Plot":
                try:
                    fig = px.box(df, y=selected_columns)
                except Exception as e:
                    return HttpResponse(f"Error creating Box Plot: {str(e)}", status=500)

            elif graph_type == "Heatmap":
                try:
                    fig = px.density_heatmap(df, x=selected_columns[0], y=selected_columns[1])
                except Exception as e:
                    return HttpResponse(f"Error creating Heatmap: {str(e)}", status=500)

            elif graph_type == "Violin Plot":
                try:
                    fig = px.violin(df, x=selected_columns[0], y=selected_columns[1], box=True, points="all")
                except Exception as e:
                    return HttpResponse(f"Error creating Violin Plot: {str(e)}", status=500)

            elif graph_type == "Sunburst Chart":
                try:
                    fig = px.sunburst(df, path=selected_columns, values=selected_columns[1])
                except Exception as e:
                    return HttpResponse(f"Error creating Sunburst Chart: {str(e)}", status=500)

            elif graph_type == "Treemap":
                try:
                    fig = px.treemap(df, path=selected_columns, values=selected_columns[1])
                except Exception as e:
                    return HttpResponse(f"Error creating Treemap: {str(e)}", status=500)

            elif graph_type == "Area Chart":
                try:
                    fig = px.area(df, x=selected_columns[0], y=selected_columns[1:])
                except Exception as e:
                    return HttpResponse(f"Error creating Area Chart: {str(e)}", status=500)

            elif graph_type == "Bubble Chart":
                try:
                    fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1], size=selected_columns[1])
                except Exception as e:
                    return HttpResponse(f"Error creating Bubble Chart: {str(e)}", status=500)

            elif graph_type == "Funnel Chart":
                try:
                    fig = px.funnel(df, x=selected_columns[0], y=selected_columns[1])
                except Exception as e:
                    return HttpResponse(f"Error creating Funnel Chart: {str(e)}", status=500)

            elif graph_type == "Density Contour":
                try:
                    fig = px.density_contour(df, x=selected_columns[0], y=selected_columns[1])
                except Exception as e:
                    return HttpResponse(f"Error creating Density Contour: {str(e)}", status=500)

            elif graph_type == "Parallel Coordinates Plot":
                try:
                    fig = px.parallel_coordinates(df, dimensions=selected_columns)
                except Exception as e:
                    return HttpResponse(f"Error creating Parallel Coordinates Plot: {str(e)}", status=500)

            elif graph_type == "Map Plot":
                try:
                    fig = px.scatter_geo(df, lat=selected_columns[0], lon=selected_columns[1])
                except Exception as e:
                    return HttpResponse(f"Error creating Map Plot: {str(e)}", status=500)

            else:
                return HttpResponse("Invalid graph type selected", status=400)

            graph_html = fig.to_html(full_html=False)
            return render(request, 'graphs.html', {'graph_html': graph_html, 'columns': df.columns.tolist()})

        except Exception as e:
            return HttpResponse(f"Unexpected error generating graph: {str(e)}", status=500)

    return HttpResponse("Invalid request method", status=400)















def graphs_view(request):
    if not request.session.get("is_logged_in"):
        return redirect("login")

    csv_data = request.session.get(SESSION_CSV_DATA)
    columns = pd.read_csv(io.StringIO(csv_data)).columns.tolist() if csv_data else []
    
    return render(request, 'graphs.html', {'columns': columns})


















import MySQLdb  
def get_mysql_connection():
    """Establish connection to MySQL database."""
    return MySQLdb.connect(
        host='127.0.0.1',
        port=3307,        
        user='rootuser',  
        password='Gangayamuna@123',  
        database='django_data_analysis',  
        charset='utf8mb4',  
        use_unicode=True
    )

import pandas as pd
import io
import MySQLdb
from django.shortcuts import render
from django.http import HttpResponse
import logging

logger = logging.getLogger(__name__)

def get_mysql_connection():
    """Establish a connection to the MySQL database."""
    try:
        conn = MySQLdb.connect(
            host="127.0.0.1",  
            user="rootuser",
            password="Gangayamuna@123",
            database="django_data_analysis",
            port=3307, 
            charset="utf8mb4"
        )
        return conn
    except MySQLdb.Error as e:
        logger.error(f"MySQL Connection Error: {e}")
        print(f"MySQL Connection Error: {e}") 
        return None


import random
import string
from django.http import JsonResponse

def generate_password():
    """Generate a random 8-character password."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8))

def save_data_to_mysql(df, table_name, primary_key_column, request):
    """Save cleaned DataFrame to MySQL with a password-protected table and log it per user."""
    if "username" not in request.session:  # Ensure user is logged in
        return HttpResponse("User not logged in. Please log in first.")

    username = request.session["username"]  # Get logged-in user's username

    columns = df.columns.tolist()
    df = df.where(pd.notna(df), None)

    conn = get_mysql_connection()
    if not conn:
        return HttpResponse("Database connection failed.")

    cursor = conn.cursor()

    cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
    result = cursor.fetchone()
    if result:
        return HttpResponse("Table already exists. Please choose another name.")

    # Create the table
    create_table_query = f"CREATE TABLE `{table_name}` ("
    create_table_query += ", ".join([f"`{col}` VARCHAR(255)" for col in columns])
    create_table_query += f", PRIMARY KEY (`{primary_key_column}`))"
    
    cursor.execute(create_table_query)

    for index, row in df.iterrows():
        row_values = [None if pd.isna(value) else value for value in row]
        insert_query = f"INSERT INTO `{table_name}` ({', '.join([f'`{col}`' for col in columns])}) VALUES ({', '.join(['%s'] * len(columns))})"
        cursor.execute(insert_query, tuple(row_values))

    # Generate and store the password
    table_password = generate_password()
    cursor.execute("CREATE TABLE IF NOT EXISTS protected_tables (table_name VARCHAR(255) PRIMARY KEY, password VARCHAR(255))")
    cursor.execute("INSERT INTO protected_tables (table_name, password) VALUES (%s, %s)", (table_name, table_password))

    conn.commit()
    cursor.close()
    conn.close()

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # Ensure logs directory exists
    user_log_file = os.path.join(log_dir, f"user_{username}_log.txt")

    with open(user_log_file, "a") as log_file:
        log_file.write(f"Table: {table_name}, Password: {table_password}\n")

    return JsonResponse({
        "message": f"Data successfully saved in table {table_name}.",
        "password": table_password,
        "log_file": user_log_file
    })


def save_in_database(request):
    """Handle the POST request to save CSV data to MySQL."""
    if request.method == 'POST':
        csv_data = request.session.get('csv_data')
        table_name = request.POST['table_name']
        primary_key = request.POST['primary_key']

        if not csv_data:
            return HttpResponse("No CSV data found in session.")

        df = pd.read_csv(io.StringIO(csv_data))

        response = save_data_to_mysql(df, table_name, primary_key, request)

        return response 
    return render(request, 'display_csv.html')


from django.shortcuts import render, HttpResponse
import MySQLdb


import re  
def analyze_sql_query(request):
    """Execute user-provided SQL query and return results, ensuring authentication runs every time."""
    sql_result = None
    sql_error = None
    columns = []
    sql_query = ""
    tables = []  # Store accessible table(s)
    table_name = ""  # Store selected table
    password = ""  # Store entered password

    if request.method == "POST":
        sql_query = request.POST.get("sql_query", "").strip()
        table_name = request.POST.get("table_name", "").strip()
        password = request.POST.get("password", "").strip()

        conn = get_mysql_connection()
        if conn:
            cursor = conn.cursor(MySQLdb.cursors.DictCursor)
            try:
                # Check if password matches a table in protected_tables
                cursor.execute("SELECT table_name FROM protected_tables WHERE password = %s", (password,))
                result = cursor.fetchone()

                if result:
                    allowed_table = result["table_name"]  # Store the allowed table
                    request.session["allowed_table"] = allowed_table  # Store it in session
                    tables = [allowed_table]  # Only allow this table to be selected
                else:
                    return HttpResponse("Invalid password. No table found.")

                if sql_query and table_name:
                    # Ensure that the user can only access the authenticated table
                    allowed_table = request.session.get("allowed_table", None)
                    if not allowed_table or table_name != allowed_table:
                        return HttpResponse("Unauthorized access to the table.")

                    # 🚨 **Extract table names from SQL query**
                    extracted_tables = extract_table_names(sql_query)

                    # 🚫 **Block query execution if it references unauthorized tables**
                    if any(tbl != allowed_table for tbl in extracted_tables):
                        return HttpResponse("Unauthorized SQL query: You can only query table '{}'.".format(allowed_table))

                    # ✅ **Execute query safely**
                    cursor.execute(sql_query)

                    if sql_query.lower().startswith("select"):
                        sql_result = cursor.fetchall()
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    else:
                        conn.commit()
                        sql_result = f"Query executed successfully: {sql_query}"

            except MySQLdb.ProgrammingError as e:
                sql_error = f"SQL Syntax Error: {str(e)}"
            except MySQLdb.DatabaseError as e:
                sql_error = f"Database Error: {str(e)}"
            except Exception as e:
                sql_error = f"Unexpected Error: {str(e)}"
            finally:
                cursor.close()
                conn.close()
        else:
            sql_error = "Database connection failed."

    return render(request, "analysis.html", {
        "sql_query": sql_query,
        "sql_result": sql_result,
        "columns": columns,
        "sql_error": sql_error,
        "tables": tables,  # Only the allowed table
        "table_name": table_name,  # Retain selected table
        "password": password  # Retain entered password
    })


def extract_table_names(sql_query):
    """Extract table names from a SQL query using regex."""
    pattern = r"(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+`?(\w+)`?"  # Match table names in SQL
    matches = re.findall(pattern, sql_query, re.IGNORECASE)
    return set(matches)  # Return unique table names


def update_sql_table(request, table_name, updated_data, primary_key_column):
    """Update existing rows in the SQL table."""
    conn = get_mysql_connection()
    if not conn:
        return HttpResponse("Database connection failed.")

    cursor = conn.cursor()

    for index, row in updated_data.iterrows():
        update_query = f"""
        UPDATE {table_name}
        SET {', '.join([f"`{col}` = %s" for col in row.index if col != primary_key_column])}
        WHERE `{primary_key_column}` = %s
        """
        values = tuple(row[col] for col in row.index if col != primary_key_column) + (row[primary_key_column],)

        try:
            cursor.execute(update_query, values)
        except MySQLdb.Error as e:
            print(f"SQL Update Error: {str(e)}")  
            return HttpResponse(f"SQL Update Error: {str(e)}")

    conn.commit()
    cursor.close()
    conn.close()
    
    return HttpResponse("Data updated successfully.")




import os

def view_saved_tables(request):
    """Display saved tables and passwords from log file"""
    log_file = "tables_log.txt"
    table_data = []

    if os.path.exists(log_file):
        with open(log_file, "r") as file:
            lines = file.readlines()[2:]  # Skip headers
            for line in lines:
                table, password = line.strip().split(" | ")
                table_data.append({"table": table, "password": password})

    return render(request, "saved_tables.html", {"table_data": table_data})


import os

def log_table_password(table_name, password):
    """Store table name and password in a log file"""
    log_file = "tables_log.txt"

    # Ensure the log file exists
    if not os.path.exists(log_file):
        with open(log_file, "w") as file:
            file.write("Table Name | Password\n")
            file.write("-" * 30 + "\n")

    with open(log_file, "a") as file:
        file.write(f"{table_name} | {password}\n")


import csv
from django.http import HttpResponse

def download_table_log(request):
    """Generate and download a CSV file of stored table passwords"""
    log_file = "tables_log.txt"
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="table_passwords.csv"'

    writer = csv.writer(response)
    writer.writerow(["Table Name", "Password"])

    if os.path.exists(log_file):
        with open(log_file, "r") as file:
            lines = file.readlines()[2:]  # Skip headers
            for line in lines:
                writer.writerow(line.strip().split(" | "))

    return response








def view_user_logs(request):
    if "username" not in request.session:
        return HttpResponse("User not logged in.")

    username = request.session["username"]
    log_file_path = os.path.join("logs", f"user_{username}_log.txt")

    if not os.path.exists(log_file_path):
        return HttpResponse("No log file found for this user.")

    with open(log_file_path, "r") as log_file:
        log_content = log_file.readlines()

    return render(request, "view_logs.html", {"log_content": log_content})

























































import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import io
from django.shortcuts import render, redirect
from django.http import HttpResponse

SESSION_DASHBOARD_DATA = "dashboard_graphs"
SESSION_CSV_DATA = "csv_data"


# def dashboard_view(request):
#     if not request.session.get("is_logged_in"):
#         return redirect("login")

#     csv_data = request.session.get(SESSION_CSV_DATA)
#     if not csv_data:
#         return HttpResponse("No CSV data found. Please upload a CSV file first.", status=400)

#     df = pd.read_csv(io.StringIO(csv_data))
#     columns = df.columns.tolist()
#     dashboard_graphs = request.session.get(SESSION_DASHBOARD_DATA, [])

#     if request.method == "POST":
#         if "delete_index" in request.POST:
#             try:
#                 delete_index = int(request.POST.get("delete_index"))
#                 if 0 <= delete_index < len(dashboard_graphs):
#                     del dashboard_graphs[delete_index]
#                     request.session[SESSION_DASHBOARD_DATA] = dashboard_graphs
#                     request.session.modified = True  # Ensure session updates
#                     return HttpResponse(status=200)
#                 return HttpResponse("Invalid index.", status=400)
#             except ValueError:
#                 return HttpResponse("Invalid request.", status=400)

#         selected_columns = request.POST.getlist('columns')
#         graph_type = request.POST.get('graph_type')
#         start_index = request.POST.get('start_index')
#         end_index = request.POST.get('end_index')
#         min_value = request.POST.get('min_value')
#         max_value = request.POST.get('max_value')

#         if not selected_columns or not graph_type:
#             return HttpResponse("Please select columns and a graph type.", status=400)

#         try:
#             start_index = int(start_index) if start_index else 0
#             end_index = int(end_index) if end_index else len(df)
#             min_value = float(min_value) if min_value else None
#             max_value = float(max_value) if max_value else None
#         except ValueError:
#             return HttpResponse("Invalid numeric filter values.", status=400)



#         df = df.iloc[start_index:end_index]  # Filter by row range


#         # Apply min/max filtering
#         if min_value is not None:
#             df = df[df[selected_columns[0]] >= min_value]
#         if max_value is not None:
#             df = df[df[selected_columns[0]] <= max_value]

#         # More flexible column selection
#         x_column = selected_columns[0] if df[selected_columns[0]].dtype == 'object' else None
#         y_columns = selected_columns if x_column is None else selected_columns[1:]

#         fig = None
#         if graph_type == "Bar Graph":
#             fig = px.bar(df, x=x_column, y=y_columns) if x_column else px.bar(df[y_columns])
#         elif graph_type == "Histogram":
#             fig = px.histogram(df, x=y_columns[0])
#         elif graph_type == "Line Chart":
#             fig = px.line(df, x=x_column, y=y_columns) if x_column else px.line(df[y_columns])
#         elif graph_type == "Scatter Plot":
#             fig = px.scatter(df, x=x_column, y=y_columns[0]) if x_column else px.scatter(df, x=y_columns[0], y=y_columns[1])
#         elif graph_type == "Pie Chart":
#             if len(y_columns) == 1 and x_column:
#                 fig = px.pie(df, names=x_column, values=y_columns[0])
#         elif graph_type == "Box Plot":
#             fig = px.box(df, y=y_columns)
#         elif graph_type == "Area Chart":
#             fig = px.area(df, x=x_column, y=y_columns) if x_column else px.area(df[y_columns])
#         elif graph_type == "Heatmap":
#             fig = ff.create_annotated_heatmap(z=df[y_columns].corr().values, x=y_columns, y=y_columns)
#         elif graph_type == "Bubble Chart":
#             if len(y_columns) == 2:
#                 fig = px.scatter(df, x=y_columns[0], y=y_columns[1], size=y_columns[1], color=y_columns[0])
#         elif graph_type == "Funnel Chart":
#             if len(y_columns) == 1 and x_column:
#                 fig = px.funnel(df, x=x_column, y=y_columns[0])
#         elif graph_type == "Violin Plot":
#             fig = px.violin(df, y=y_columns, box=True, points="all")
#         elif graph_type == "Density Contour":
#             if len(y_columns) >= 2:
#                 fig = px.density_contour(df, x=y_columns[0], y=y_columns[1])
#         elif graph_type == "Parallel Coordinates Plot":
#             fig = px.parallel_coordinates(df, dimensions=y_columns)
#         elif graph_type == "Sunburst Chart":
#             if len(y_columns) == 1 and x_column:
#                 fig = px.sunburst(df, path=[x_column], values=y_columns[0])
#         elif graph_type == "Map Plot":
#             if 'latitude' in df.columns and 'longitude' in df.columns:
#                 fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", zoom=3, mapbox_style="carto-positron")

#         if fig:
#             graph_html = fig.to_html(full_html=False)
#             dashboard_graphs.append(graph_html)
#             request.session[SESSION_DASHBOARD_DATA] = dashboard_graphs
#             request.session.modified = True  # Ensure session updates

#     return render(request, "dashboard.html", {"columns": columns, "graph_htmls": dashboard_graphs})


def dashboard_view(request):
    if not request.session.get("is_logged_in"):
        return redirect("login")

    # Retrieve CSV data
    csv_data = request.session.get(SESSION_CSV_DATA)
    if not csv_data:
        return HttpResponse("No CSV data found. Please upload a CSV file first.", status=400)

    try:
        df = pd.read_csv(io.StringIO(csv_data))
    except pd.errors.EmptyDataError:
        return HttpResponse("The uploaded CSV file is empty.", status=400)
    except pd.errors.ParserError:
        return HttpResponse("Error parsing the CSV file. Please check the file format.", status=400)
    except Exception as e:
        return HttpResponse(f"Unexpected error reading CSV file: {str(e)}", status=500)

    columns = df.columns.tolist()
    dashboard_graphs = request.session.get(SESSION_DASHBOARD_DATA, [])

    if request.method == "POST":
        if "delete_index" in request.POST:
            try:
                delete_index = int(request.POST.get("delete_index"))
                if 0 <= delete_index < len(dashboard_graphs):
                    del dashboard_graphs[delete_index]
                    request.session[SESSION_DASHBOARD_DATA] = dashboard_graphs
                    request.session.modified = True  # Ensure session updates
                    return HttpResponse(status=200)
                return HttpResponse("Invalid index.", status=400)
            except ValueError:
                return HttpResponse("Invalid request. Index must be an integer.", status=400)

        # Get user selections
        selected_columns = request.POST.getlist("columns")
        graph_type = request.POST.get("graph_type")
        start_index = request.POST.get("start_index")
        end_index = request.POST.get("end_index")
        min_value = request.POST.get("min_value")
        max_value = request.POST.get("max_value")

        if not selected_columns or not graph_type:
            return HttpResponse("Please select at least one column and a graph type.", status=400)

        # Convert inputs safely
        try:
            start_index = int(start_index) if start_index else 0
            end_index = int(end_index) if end_index else len(df)
            min_value = float(min_value) if min_value else None
            max_value = float(max_value) if max_value else None
        except ValueError:
            return HttpResponse("Invalid numeric filter values.", status=400)

        # Validate column selection
        invalid_columns = [col for col in selected_columns if col not in df.columns]
        if invalid_columns:
            return HttpResponse(f"Invalid column selection: {', '.join(invalid_columns)}", status=400)

        df = df.iloc[start_index:end_index]  # Filter by row range

        # Apply min/max filtering
        try:
            if min_value is not None:
                df = df[df[selected_columns[0]] >= min_value]
            if max_value is not None:
                df = df[df[selected_columns[0]] <= max_value]
        except KeyError:
            return HttpResponse("Error filtering data. Ensure the selected column exists.", status=400)
        except TypeError:
            return HttpResponse("Filtering error: Ensure numerical filters are applied to numeric columns.", status=400)

        # Determine X and Y axis columns
        x_column = selected_columns[0] if df[selected_columns[0]].dtype == "object" else None
        y_columns = selected_columns if x_column is None else selected_columns[1:]

        fig = None
        try:
            if graph_type == "Bar Graph":
                fig = px.bar(df, x=x_column, y=y_columns) if x_column else px.bar(df[y_columns])
            elif graph_type == "Histogram":
                fig = px.histogram(df, x=y_columns[0])
            elif graph_type == "Line Chart":
                fig = px.line(df, x=x_column, y=y_columns) if x_column else px.line(df[y_columns])
            elif graph_type == "Scatter Plot":
                fig = px.scatter(df, x=x_column, y=y_columns[0]) if x_column else px.scatter(df, x=y_columns[0], y=y_columns[1])
            elif graph_type == "Pie Chart":
                if len(y_columns) == 1 and x_column:
                    fig = px.pie(df, names=x_column, values=y_columns[0])
            elif graph_type == "Box Plot":
                fig = px.box(df, y=y_columns)
            elif graph_type == "Area Chart":
                fig = px.area(df, x=x_column, y=y_columns) if x_column else px.area(df[y_columns])
            elif graph_type == "Heatmap":
                fig = ff.create_annotated_heatmap(z=df[y_columns].corr().values, x=y_columns, y=y_columns)
            elif graph_type == "Bubble Chart":
                if len(y_columns) == 2:
                    fig = px.scatter(df, x=y_columns[0], y=y_columns[1], size=y_columns[1], color=y_columns[0])
            elif graph_type == "Funnel Chart":
                if len(y_columns) == 1 and x_column:
                    fig = px.funnel(df, x=x_column, y=y_columns[0])
            elif graph_type == "Violin Plot":
                fig = px.violin(df, y=y_columns, box=True, points="all")
            elif graph_type == "Density Contour":
                if len(y_columns) >= 2:
                    fig = px.density_contour(df, x=y_columns[0], y=y_columns[1])
            elif graph_type == "Parallel Coordinates Plot":
                fig = px.parallel_coordinates(df, dimensions=y_columns)
            elif graph_type == "Sunburst Chart":
                if len(y_columns) == 1 and x_column:
                    fig = px.sunburst(df, path=[x_column], values=y_columns[0])
            elif graph_type == "Map Plot":
                if "latitude" in df.columns and "longitude" in df.columns:
                    fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", zoom=3, mapbox_style="carto-positron")
            else:
                return HttpResponse("Invalid graph type selected.", status=400)

            if fig:
                graph_html = fig.to_html(full_html=False)
                dashboard_graphs.append(graph_html)
                request.session[SESSION_DASHBOARD_DATA] = dashboard_graphs
                request.session.modified = True  # Ensure session updates

        except Exception as e:
            return HttpResponse(f"Unexpected error generating graph: {str(e)}", status=500)

    return render(request, "dashboard.html", {"columns": columns, "graph_htmls": dashboard_graphs})


import pandas as pd
import io
import numpy as np
import base64
import plotly.express as px
from django.shortcuts import render, redirect
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

SESSION_CSV_DATA = "csv_data"

def advanced_preprocessing(request):
    if not request.session.get("is_logged_in"):
        return redirect("login")

    csv_data = request.session.get(SESSION_CSV_DATA)
    columns = []
    plot_html = None

    if not csv_data:
        return render(request, "advanced_preprocessing.html", {
            "columns": columns,
            "error": "No CSV data found in session. Please upload a file first."
        })

    df = pd.read_csv(io.StringIO(csv_data))
    columns = df.columns.tolist()

    if request.method == "POST":
        method = request.POST.get("method")
        column_name = request.POST.get("column")

        if not column_name or column_name not in df.columns:
            return render(request, "advanced_preprocessing.html", {
                "columns": columns,
                "error": f"Invalid column selection: {column_name}. Please choose a valid column."
            })

        try:
            # ✅ **Handling Outliers**
            if method == "remove_outliers_iqr":
                Q1 = df[column_name].quantile(0.25)
                Q3 = df[column_name].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[column_name] >= Q1 - 1.5 * IQR) & (df[column_name] <= Q3 + 1.5 * IQR)]

            elif method == "remove_outliers_zscore":
                mean = df[column_name].mean()
                std_dev = df[column_name].std()
                df = df[(df[column_name] >= mean - 3 * std_dev) & (df[column_name] <= mean + 3 * std_dev)]

            # ✅ **Feature Engineering**
            elif method == "log_transform":
                df[column_name] = np.log1p(df[column_name])

            elif method == "square_transform":
                df[column_name] = df[column_name] ** 2

            elif method == "polynomial_features":
                poly = PolynomialFeatures(degree=2, include_bias=False)
                transformed_data = poly.fit_transform(df[[column_name]])
                df = pd.concat([df, pd.DataFrame(transformed_data, columns=[f"{column_name}_poly_{i}" for i in range(transformed_data.shape[1])])], axis=1)

            elif method == "binning":
                df[column_name + "_bin"] = pd.cut(df[column_name], bins=4, labels=["Low", "Medium", "High", "Very High"])

            # ✅ **Scaling**
            elif method == "standard_scaling":
                df[column_name] = (df[column_name] - df[column_name].mean()) / df[column_name].std()

            elif method == "min_max_scaling":
                df[column_name] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())

            # ✅ **Handling Missing Values**
            elif method == "fill_missing_mean":
                imputer = SimpleImputer(strategy="mean")
                df[column_name] = imputer.fit_transform(df[[column_name]])

            elif method == "fill_missing_knn":
                imputer = KNNImputer(n_neighbors=3)
                df[column_name] = imputer.fit_transform(df[[column_name]])

            elif method == "fill_missing_gboost":
                model = GradientBoostingRegressor()
                train_data = df.dropna(subset=[column_name])
                test_data = df[df[column_name].isnull()]
                if not test_data.empty:
                    model.fit(train_data.drop(columns=[column_name]), train_data[column_name])
                    df.loc[df[column_name].isnull(), column_name] = model.predict(test_data.drop(columns=[column_name]))

            elif method == "fill_missing_rf":
                model = RandomForestRegressor()
                train_data = df.dropna(subset=[column_name])
                test_data = df[df[column_name].isnull()]
                if not test_data.empty:
                    model.fit(train_data.drop(columns=[column_name]), train_data[column_name])
                    df.loc[df[column_name].isnull(), column_name] = model.predict(test_data.drop(columns=[column_name]))

            # ✅ **Encoding Categorical Features**
            elif method == "one_hot_encoding":
                df = pd.get_dummies(df, columns=[column_name])

            elif method == "label_encoding":
                encoder = LabelEncoder()
                df[column_name] = encoder.fit_transform(df[column_name])

            # ✅ **Text Processing**
            elif method == "tfidf_vectorization":
                vectorizer = TfidfVectorizer(max_features=100)
                tfidf_matrix = vectorizer.fit_transform(df[column_name].astype(str))
                df = pd.concat([df, pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])], axis=1)

            # ✅ **Data Balancing (SMOTE)**
            elif method == "smote":
                smote = SMOTE()
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                if len(num_cols) > 1:
                    df_resampled, _ = smote.fit_resample(df[num_cols], df[num_cols[0]])
                    df = pd.DataFrame(df_resampled, columns=num_cols)

            # ✅ **Save Processed Data**
            request.session[SESSION_CSV_DATA] = df.to_csv(index=False)

            # ✅ **Generate Plot**
            if column_name in df.columns:
                fig = px.box(df, y=column_name, title=f"Boxplot After {method} on {column_name}")
                plot_html = fig.to_html(full_html=False)

        except Exception as e:
            return render(request, "advanced_preprocessing.html", {
                "columns": columns,
                "error": f"An error occurred: {str(e)}"
            })

        return render(request, "advanced_preprocessing.html", {
            "columns": columns,
            "method": method,
            "column_name": column_name,
            "plot_html": plot_html
        })

    return render(request, "advanced_preprocessing.html", {"columns": columns})



import pandas as pd
import io
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np

# Additional Libraries
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# XGBoost and LightGBM (make sure you have them installed!)
try:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    xgboost_available = True
except ImportError:
    xgboost_available = False





def predictive_analysis(request):
    """Handles ML & Predictive Analysis on uploaded CSV data"""

    csv_data = request.session.get("csv_data")

    if not csv_data:
        return render(request, "predictive_analysis.html", {"error": "No CSV data found!"})

    df = pd.read_csv(io.StringIO(csv_data))

    # Drop non-numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if len(numeric_cols) < 2:
        return render(request, "predictive_analysis.html", {"error": "Need at least 2 numeric columns for ML!"})

    # Get user-selected target column
    selected_column = request.GET.get("target_column")
    if selected_column not in numeric_cols:
        selected_column = numeric_cols[-1]  # Default to last numeric column

    # Get user-selected ML model
    selected_model = request.GET.get("model", "Linear Regression")

    # Get index selection method
    index_selection = request.GET.get("index_selection", "all")

    # Filter correlated features
    correlation_matrix = df[numeric_cols].corr()
    correlated_features = correlation_matrix[selected_column][correlation_matrix[selected_column].abs() > 0.1].index.tolist()

    if len(correlated_features) < 2:
        return render(request, "predictive_analysis.html", {"error": "Not enough correlated features for ML!"})

    X = df[correlated_features].drop(columns=[selected_column])  # Features
    y = df[selected_column]  # Target column

    # Handle missing values using Mean Imputation
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Standardize the Features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Handle Index Selection
    if index_selection == "all":
        pass  # Use entire dataset
    elif index_selection == "range":
        start_index = int(request.GET.get("start_index", 0))
        end_index = int(request.GET.get("end_index", len(X)))
        X = X[start_index:end_index]
        y = y[start_index:end_index]

    # Train ML Models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Support Vector Regression (SVR)": SVR(),
    }

    if xgboost_available:
        models["XGBoost"] = XGBRegressor()
        models["LightGBM"] = LGBMRegressor()

    if selected_model in models:
        model = models[selected_model]
        model.fit(X_train, y_train)
        df[f"Predicted {selected_column}"] = model.predict(X)  # Add Predictions to DataFrame
    else:
        df[f"Predicted {selected_column}"] = None

    # K-Means Clustering (Optional)
    if request.GET.get("kmeans") == "yes":
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X)
    else:
        df["Cluster"] = None

    # Anomaly Detection (Optional)
    if request.GET.get("anomaly_detection") == "yes":
        isolation_forest = IsolationForest(contamination=0.05, random_state=42)
        df["Anomaly"] = isolation_forest.fit_predict(X)
    else:
        df["Anomaly"] = None

    # Convert DataFrame to HTML Table
    table_html = df.to_html(classes="table table-bordered", index=False)

    return render(request, "predictive_analysis.html", {
        "message": "ML Models trained successfully!",
        "table_html": table_html,  # Pass full table
        "columns": numeric_cols,
        "selected_column": selected_column,
        "model_options": models.keys(),
        "selected_model": selected_model,
        "index_selection": index_selection
    })
