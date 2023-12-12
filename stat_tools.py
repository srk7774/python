# helper function for viewing pandas dataframes
def glimpse(df, max_width=76):

    # find the max string lengths of the column names and dtypes for formatting
    _max_len = max([len(col) for col in df])
    _max_dtype_label_len = max([len(str(df[col].dtype)) for col in df])

    # print the dimensions of the dataframe
    print(f"{type(df)}:  {df.shape[0]} rows of {df.shape[1]} columns")

    # print the name, dtype and first few values of each column
    for _column in df:

        _col_vals = df[_column].head(max_width).to_list()
        _col_type = str(df[_column].dtype)

        output_col = f"{_column}:".ljust(_max_len+1, ' ')
        output_dtype = f" {_col_type}".ljust(_max_dtype_label_len+3, ' ')

        output_combined = f"{output_col} {output_dtype} {_col_vals}"

        # trim the output if too long
        if len(output_combined) > max_width:
            output_combined = output_combined[0:(max_width-4)] + " ..."

        print(output_combined)


def omega_sq_partial(df_model, F, N):
    """
    Calculate the custom statistic based on the given formula.
    
    Parameters:
    df_model (float): Degrees of freedom for the model.
    MS_model (float): Mean square for the model.
    MS_error (float): Mean square for the error.
    SS_model (float): Sum of squares for the model.
    N (int): Total number of observations.
    
    Returns:
    float: The calculated value of the custom statistic.
    """
    # Calculate the statistic according to the provided formula
    numerator = df_model * (F - 1)
    denominator = (df_model * (F - 1)) + N

    # Handling division by zero
    if denominator == 0:
        return float('inf')  # Return infinity if denominator is zero

    return numerator / denominator
