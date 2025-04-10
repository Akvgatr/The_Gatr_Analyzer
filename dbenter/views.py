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
import numpy as np
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
        add_zero_column = request.POST.get("add_zero_column")
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
            
            if add_zero_column:
                df["Zero Column"] = 0.0  # Add a column where every row is 0



















        #     # Handle Search
        #     if search_query and column_name in df.columns:
        #         df = df[df[column_name].astype(str).str.contains(search_query, case=False, na=False)]


        #     # Handle Filter
        #     if request.method == "POST":
        #         action = request.POST.get("action")

        #     if action == "filter":
        #          filter_column = request.POST.get("filter_column")
        #          filter_value = request.POST.get("filter_value")
        #          filter_condition = request.POST.get("filter_condition")  # Ensure this is set in your form

        #     if filter_col1 and filter_col2 and filter_op:
        #        if filter_col1 in df.columns and filter_col2 in df.columns:
        #         try:
        #             if filter_op == "add":
        #                 df["temp_filter_col"] = df[filter_col1] + df[filter_col2]
        #             elif filter_op == "subtract":
        #                 df["temp_filter_col"] = df[filter_col1] - df[filter_col2]
        #             elif filter_op == "multiply":
        #                 df["temp_filter_col"] = df[filter_col1] * df[filter_col2]
        #             elif filter_op == "divide":
        #                 df["temp_filter_col"] = df[filter_col1] / df[filter_col2]

        #             # Apply filtering on the temporary column
        #             if filter_condition and filter_value:
        #                 filter_value = float(filter_value) if filter_value.replace('.', '', 1).isdigit() else filter_value

        #                 if filter_condition == "=":
        #                     df = df[df["temp_filter_col"] == filter_value]
        #                 elif filter_condition == ">":
        #                     df = df[df["temp_filter_col"] > filter_value]
        #                 elif filter_condition == "<":
        #                     df = df[df["temp_filter_col"] < filter_value]
        #                 elif filter_condition == ">=":
        #                     df = df[df["temp_filter_col"] >= filter_value]
        #                 elif filter_condition == "<=":
        #                     df = df[df["temp_filter_col"] <= filter_value]
        #                 elif filter_condition == "!=":
        #                     df = df[df["temp_filter_col"] != filter_value]
        #             df.drop(columns=["temp_filter_col"], inplace=True)  # Remove temporary column

        #         except Exception as e:
        #             messages.error(request, f"Error in column operation: {e}")

        # # Handle numeric filtering
        # if filter_column and filter_value and filter_column in df.columns:
        #     try:
        #         filter_value = float(filter_value) if filter_value.replace('.', '', 1).isdigit() else filter_value

        #         if filter_condition == "=":
        #             df = df[df[filter_column] == filter_value]
        #         elif filter_condition == ">":
        #             df = df[df[filter_column] > filter_value]
        #         elif filter_condition == "<":
        #             df = df[df[filter_column] < filter_value]
        #         elif filter_condition == ">=":
        #             df = df[df[filter_column] >= filter_value]
        #         elif filter_condition == "<=":
        #             df = df[df[filter_column] <= filter_value]
        #         elif filter_condition == "!=":
        #             df = df[df[filter_column] != filter_value]
        #     except Exception as e:
        #         messages.error(request, f"Error in filtering: {e}")




        # # Handle text filtering
        # elif action == "text_filter":
        #     column_name = request.POST.get("column_name")
        #     search_query = request.POST.get("search_query")

        #     if column_name and search_query and column_name in df.columns:
        #         try:
        #             df = df[df[column_name].astype(str).str.contains(search_query, case=False, na=False)]
        #         except Exception as e:
        #             messages.error(request, f"Error in text filtering: {e}")


# Handle Search
        if search_query and column_name in df.columns:
            df = df[df[column_name].astype(str).str.contains(search_query, case=False, na=False)]

        # Handle Filter
        if request.method == "POST":
            action = request.POST.get("action")

            # New: Range Filtering
            if action == "range_filter":
                range_column = request.POST.get("range_column")
                range_min = request.POST.get("range_min")
                range_max = request.POST.get("range_max")
                
                # Row selection range
                row_start = request.POST.get("row_start")
                row_end = request.POST.get("row_end")

                if range_column and range_column in df.columns:
                    try:
                        # Convert range min and max to float if they are numeric
                        range_min = float(range_min) if range_min and range_min.replace('.', '', 1).isdigit() else None
                        range_max = float(range_max) if range_max and range_max.replace('.', '', 1).isdigit() else None

                        # Filter by column value range
                        if range_min is not None and range_max is not None:
                            df = df[(df[range_column] >= range_min) & (df[range_column] <= range_max)]
                        elif range_min is not None:
                            df = df[df[range_column] >= range_min]
                        elif range_max is not None:
                            df = df[df[range_column] <= range_max]

                        # Row selection range
                        row_start = int(row_start) if row_start and row_start.isdigit() else None
                        row_end = int(row_end) if row_end and row_end.isdigit() else None

                        if row_start is not None and row_end is not None:
                            df = df.iloc[row_start:row_end+1]
                        elif row_start is not None:
                            df = df.iloc[row_start:]
                        elif row_end is not None:
                            df = df.iloc[:row_end+1]

                    except Exception as e:
                        messages.error(request, f"Error in range filtering: {e}")

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
                                    elif filter_condition == "!=":
                                        df = df[df["temp_filter_col"] != filter_value]
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
                        elif filter_condition == "!=":
                            df = df[df[filter_column] != filter_value]
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

def add_zero_column(df):
    df['Zero Column'] = 0  # Add a new column with every row set to 0
    return df


import json
from django.core.exceptions import ValidationError

import pandas as pd
from django.core.exceptions import ValidationError
from .models import CleanedData  # Ensure this model is correctly defined













def apply_cleaning_method(df, method, column_name=None, **kwargs):
    import pandas as pd  # Add this import at the beginning of the function
    import numpy as np   # Also import numpy since you use np.nan in some methods
     
    if method == 'DropCols':
        df.drop(columns=column_name, inplace=True)


    elif method == 'RemovePunctuation':
        # Remove all punctuation from a column
        df[column_name] = df[column_name].astype(str).str.replace(r'[^\w\s]', '', regex=True)
        print(f"\nRemoved punctuation from {column_name}")

    elif method == 'RemoveAlphabets':
        # Create a new column name
        new_column = f"{column_name}_noalphabets"

        # Remove all alphabets, keep numbers and punctuation
        df[new_column] = df[column_name].astype(str).str.replace(r'[a-zA-Z]', '', regex=True)
        
        print(f"\nRemoved alphabets and stored in new column {new_column}")

    elif method == 'AddTrailingSpace':
        # Create a new column name
        new_column = f"{column_name}_spaced"

        # Add a trailing space to each value
        df[new_column] = df[column_name].astype(str) + " "

        print(f"\nAdded trailing space and stored in new column '{new_column}'")

    elif method == 'TextLength':
        # Calculate length of text in a column
        new_column = kwargs.get('new_column', f'{column_name}_length')
        df[new_column] = df[column_name].astype(str).str.len()
        print(f"\nAdded length column for {column_name}")

    elif method == 'Normalize':
        # Check if column is not numeric and attempt conversion
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            try:
                # Attempt to convert to numeric
                df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
                print(f"\nConverted {column_name} to numeric")
            except Exception as e:
                print(f"Could not convert {column_name} to numeric: {e}")
                # continue  # Skip to next iteration if conversion fails
        
        # Normalize numeric column (min-max scaling)
        min_val = df[column_name].min()
        max_val = df[column_name].max()
        
        # Check if min and max are different to avoid division by zero
        if min_val != max_val:
            df[column_name] = (df[column_name] - min_val) / (max_val - min_val)
            print(f"\nNormalized {column_name}")
        else:
            print(f"Cannot normalize {column_name} - all values are the same")


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

    elif method == 'UppercaseFirst':
        # Capitalize the first letter of each string in the column
        df[column_name] = df[column_name].astype(str).apply(
            lambda x: x[0].upper() + x[1:] if len(x) > 0 else x
        )

    elif method == 'LowercaseFirst':
        # Lowercase the first letter of each string in the column
        df[column_name] = df[column_name].astype(str).apply(
            lambda x: x[0].lower() + x[1:] if len(x) > 0 else x
        )
    
    elif method == 'Uppercase':
        # Convert the entire column to uppercase
        df[column_name] = df[column_name].astype(str).str.upper()
    
    elif method == 'Lowercase':
        # Convert the entire column to lowercase
        df[column_name] = df[column_name].astype(str).str.lower()
    
    elif method == 'RemoveSpaces':
        # Remove all spaces from the column
        df[column_name] = df[column_name].astype(str).str.replace(' ', '', regex=False)
    
    elif method == 'TrimSpaces':
        # Remove leading and trailing spaces
        df[column_name] = df[column_name].astype(str).str.strip()




    elif method == 'ExtractDate':
        import pandas as pd
        
        try:
            # Try to convert the column to datetime format
            datetime_series = pd.to_datetime(df[column_name], errors='coerce')
            
            # Create a new column with just the date part (day of month)
            new_column = f"{column_name}_day"
            df[new_column] = datetime_series.dt.day
            print(f"\nExtracted day of month to column '{new_column}'")
        except Exception as e:
            print(f"Error extracting date: {e}")
    
    elif method == 'ExtractMonth':
        import pandas as pd
        
        try:
            # Try to convert the column to datetime format
            datetime_series = pd.to_datetime(df[column_name], errors='coerce')
            
            # Create a new column with just the month
            new_column = f"{column_name}_month"
            df[new_column] = datetime_series.dt.month
            
            # Optionally, create a month name column
            month_name_column = f"{column_name}_month_name"
            df[month_name_column] = datetime_series.dt.strftime('%B')  # Full month name
            
            print(f"\nExtracted month to columns '{new_column}' (numeric) and '{month_name_column}' (name)")
        except Exception as e:
            print(f"Error extracting month: {e}")
    
    elif method == 'ExtractYear':
        import pandas as pd
        
        try:
            # Try to convert the column to datetime format
            datetime_series = pd.to_datetime(df[column_name], errors='coerce')
            
            # Create a new column with just the year
            new_column = f"{column_name}_year"
            df[new_column] = datetime_series.dt.year
            print(f"\nExtracted year to column '{new_column}'")
        except Exception as e:
            print(f"Error extracting year: {e}")











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
                # Convert to numeric where possible for arithmetic operations
                if operation in ['+', '-', '*', '/']:
                    df[col1] = pd.to_numeric(df[col1], errors='coerce')
                    df[col2] = pd.to_numeric(df[col2], errors='coerce')
                
                if operation == '+':
                    df[new_col] = df[col1] + df[col2]
                elif operation == '-':
                    df[new_col] = df[col1] - df[col2]
                elif operation == '*':
                    df[new_col] = df[col1] * df[col2]
                elif operation == '/':
                    # Avoid division by zero
                    df[new_col] = df[col1].div(df[col2], fill_value=0)
                
                # Replace NaN with empty string after operation
                df[new_col] = df[new_col].fillna('')
                
            except Exception as e:
                raise ValidationError(f"Error performing operation: {e}")

    elif method == 'DropNullRows':
        df.dropna(inplace=True)

    elif method == 'FillnanMean':
        # Fill NaNs with mean for numeric columns
        df = df.copy()
        if column_name is None:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
            print("\nFilled NaN values with mean across all numeric columns")
        else:
            if pd.api.types.is_numeric_dtype(df[column_name]):
                df[column_name] = df[column_name].fillna(df[column_name].mean())
                print(f"\nFilled NaN values in {column_name} with column mean")
            else:
                print(f"\nColumn {column_name} is not numeric. Cannot fill with mean.")
        return df

    elif method == 'FillnanMedian':
        # Fill NaNs with median for numeric columns
        df = df.copy()
        if column_name is None:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
            print("\nFilled NaN values with median across all numeric columns")
        else:
            if pd.api.types.is_numeric_dtype(df[column_name]):
                df[column_name] = df[column_name].fillna(df[column_name].median())
                print(f"\nFilled NaN values in {column_name} with column median")
            else:
                print(f"\nColumn {column_name} is not numeric. Cannot fill with median.")
        return df

    elif method == 'FillnanMode':
        # Fill NaNs with mode for both numeric and object columns
        df = df.copy()
        if column_name is None:
            for col in df.columns:
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
                df[col] = df[col].fillna(mode_val)
            print("\nFilled NaN values with mode across all columns")
        else:
            mode_val = df[column_name].mode().iloc[0] if not df[column_name].mode().empty else np.nan
            df[column_name] = df[column_name].fillna(mode_val)
            print(f"\nFilled NaN values in {column_name} with column mode")
        return df

    elif method == 'FillnanBfill':
        # Fill NaNs using backward fill method
        df = df.copy()
        if column_name is None:
            df = df.fillna(method='bfill')
            print("\nFilled NaN values using backward fill across all columns")
        else:
            df[column_name] = df[column_name].fillna(method='bfill')
            print(f"\nFilled NaN values in {column_name} using backward fill")
        return df

    elif method == 'FillnanForwardfill':
        # Fill NaNs using forward fill method
        df = df.copy()
        if column_name is None:
            df = df.fillna(method='ffill')
            print("\nFilled NaN values using forward fill across all columns")
        else:
            df[column_name] = df[column_name].fillna(method='ffill')
            print(f"\nFilled NaN values in {column_name} using forward fill")
        return df

    elif method == 'FillnanMostFrequent':
        # Fill NaNs with the most frequent value
        df = df.copy()
        if column_name is None:
            for col in df.columns:
                most_frequent = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
                df[col] = df[col].fillna(most_frequent)
            print("\nFilled NaN values with most frequent value across all columns")
        else:
            most_frequent = df[column_name].mode().iloc[0] if not df[column_name].mode().empty else np.nan
            df[column_name] = df[column_name].fillna(most_frequent)
            print(f"\nFilled NaN values in {column_name} with most frequent value")
        return df

    # Rest of the previous methods remain the same...


    df.fillna('', inplace=True)  # Final step to replace remaining NaN with empty strings

    cleaned_data = df.to_dict(orient='records')
    CleanedData.objects.all().delete()
    CleanedData.objects.create(cleaned_data=cleaned_data)

    return df



































def ConvertDataType(df, column_name, target_type='str', **kwargs):
    """
    Converts a specified column in a DataFrame to a target data type.
    
    Parameters:
    - df: pandas DataFrame
    - column_name: Name of the column to convert
    - target_type: Target data type ('str', 'int', 'float', 'datetime')
    - **kwargs: Additional parameters for specific conversions
    
    Returns:
    - Modified DataFrame with the column converted to the target type
    """
    try:
        # Remove any leading/trailing whitespace before conversion
        df[column_name] = df[column_name].astype(str).str.strip()
        
        if target_type == 'str':
            df[column_name] = df[column_name].astype(str)
        
        elif target_type == 'int':
            # Handle various potential input formats
            df[column_name] = pd.to_numeric(
                df[column_name], 
                errors='coerce',  # Converts problematic values to NaN
                downcast='integer'  # Tries to use the smallest possible integer type
            ).astype('Int64')  # Use nullable integer type to handle NaN values
        
        elif target_type == 'float':
            df[column_name] = pd.to_numeric(
                df[column_name], 
                errors='coerce'
            )
        
        elif target_type == 'datetime':
            # Multiple date parsing options
            df[column_name] = pd.to_datetime(
                df[column_name], 
                errors='coerce',
                infer_datetime_format=True,
                format=kwargs.get('format', None)  # Optional custom format
            )
        
        print(f"\nSuccessfully converted {column_name} to {target_type}")
        print(f"New column dtype: {df[column_name].dtype}")
        
    except Exception as e:
        print(f"Error converting {column_name} to {target_type}: {e}")
        raise

    return df





def drop_rows_equal(df, column_name, expr, use_numeric):
    return df[df[column_name] == expr].index

def drop_rows_not_equal(df, column_name, expr, use_numeric):
    return df[df[column_name] != expr].index

def drop_rows_greater(df, column_name, expr, use_numeric):
    return df[df[column_name] > expr].index

def drop_rows_less(df, column_name, expr, use_numeric):
    return df[df[column_name] < expr].index

def drop_rows_greater_equal(df, column_name, expr, use_numeric):
    return df[df[column_name] >= expr].index

def drop_rows_less_equal(df, column_name, expr, use_numeric):
    return df[df[column_name] <= expr].index

def drop_rows_contains(df, column_name, expr, use_numeric):
    return df[df[column_name].astype(str).str.contains(str(expr), na=False)].index

def drop_rows_startswith(df, column_name, expr, use_numeric):
    return df[df[column_name].astype(str).str.startswith(str(expr), na=False)].index

def drop_rows_endswith(df, column_name, expr, use_numeric):
    return df[df[column_name].astype(str).str.endswith(str(expr), na=False)].index







def apply_row_filter(df, column_name, condition, expr, use_numeric=False):
    """Apply the appropriate row filter based on the condition."""
    
    condition_map = {
        '==': drop_rows_equal,
        '!=': drop_rows_not_equal,
        '>': drop_rows_greater,
        '<': drop_rows_less,
        '>=': drop_rows_greater_equal,
        '<=': drop_rows_less_equal,
        'contains': drop_rows_contains,
        'startswith': drop_rows_startswith,
        'endswith': drop_rows_endswith
    }

    if condition in condition_map:
        return condition_map[condition](df, column_name, expr, use_numeric)
    
    # If no valid condition is found, return an empty index
    return pd.Index([])










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

















