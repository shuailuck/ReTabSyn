import pandas as pd
import numpy as np
from dython.nominal import theils_u
import seaborn as sns
import matplotlib.pyplot as plt

def detect_column_types(dataframe):
    continuous_columns = []
    categorical_columns = []

    for col in dataframe.columns:
        # Calculate the ratio of unique values to the total number of rows
        n_unique = dataframe[col].nunique()

        # If the ratio is below the threshold, consider the column as categorical
        if n_unique >= 2 and n_unique <= 25 and dataframe[col].dtype == 'int64':
            categorical_columns.append(col)
        elif dataframe[col].dtype == 'object':
            categorical_columns.append(col)
        else:
            continuous_columns.append(col)

    return continuous_columns, categorical_columns

def theils_u_mat(df):
  # Compute Theil's U-statistics between each pair of columns
  cate_columns = df.shape[1]
  theils_u_mat = np.zeros((cate_columns, cate_columns))

  for i in range(cate_columns):
      for j in range(cate_columns):
          theils_u_mat[i, j] = theils_u(df.iloc[:, i], df.iloc[:, j])

  return theils_u_mat

# See the post https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def correlation_ratio(categories, measurements):
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat)+1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0,cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
        numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
        denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = numerator/denominator
        return eta

def ratio_mat(df, continuous_columns, categorical_columns):
    rat_mat = pd.DataFrame(index=continuous_columns, columns=categorical_columns)
    
    if len(categorical_columns) == 0 or len(continuous_columns) == 0:
        return np.zeros(1)
    else:
        for cat_col in categorical_columns:
            for cont_col in continuous_columns:
                rat_mat.loc[cont_col, cat_col] = correlation_ratio(df[cat_col], df[cont_col])
    return rat_mat.astype(float).values

def fillNa_cont(df):
    for col in df.columns:
        mean_values = df[col].mean()
        df[col] = df[col].fillna(mean_values)
    return df

def fillNa_cate(df):
    for col in df.columns:
        mode_values = df[col].mode()[0]
        df[col] = df[col].fillna(mode_values)
    return df

def compute_correlation(df, continuous_columns, categorical_columns):

    num_mat = pd.DataFrame(df[continuous_columns])
    cat_mat = pd.DataFrame(df[categorical_columns])
    
    num_mat = fillNa_cont(num_mat)
    cat_mat = fillNa_cate(cat_mat)
    
    pearson_sub_matrix = np.corrcoef(num_mat, rowvar = False)
    theils_u_matrix = theils_u_mat(cat_mat)
    correl_ratio_mat = ratio_mat(df, continuous_columns, categorical_columns)
   
    return (pearson_sub_matrix, theils_u_matrix, correl_ratio_mat)

def create_combined_correlation_matrix(df):
    """
    Create a single correlation matrix combining all columns using appropriate correlation methods.
    
    Args:
        df (pd.DataFrame): Input dataframe with mixed numerical and categorical columns
    Returns:
        pd.DataFrame: Combined correlation matrix
    """
    continuous_columns, categorical_columns = detect_column_types(df)
    all_columns = continuous_columns + categorical_columns
    n_cols = len(all_columns)
    
    # Initialize the combined correlation matrix
    combined_corr = pd.DataFrame(np.zeros((n_cols, n_cols)), 
                               index=all_columns, 
                               columns=all_columns)
    
    # Fill numerical-numerical correlations (Pearson)
    if len(continuous_columns) > 0:
        num_mat = pd.DataFrame(df[continuous_columns])
        num_mat = fillNa_cont(num_mat)
        pearson_corr = np.corrcoef(num_mat, rowvar=False)
        for i, col1 in enumerate(continuous_columns):
            for j, col2 in enumerate(continuous_columns):
                combined_corr.loc[col1, col2] = pearson_corr[i, j]
    
    # Fill categorical-categorical correlations (Theil's U)
    if len(categorical_columns) > 0:
        cat_mat = pd.DataFrame(df[categorical_columns])
        cat_mat = fillNa_cate(cat_mat)
        theils_matrix = theils_u_mat(cat_mat)
        for i, col1 in enumerate(categorical_columns):
            for j, col2 in enumerate(categorical_columns):
                combined_corr.loc[col1, col2] = theils_matrix[i, j]
    
    # Fill numerical-categorical correlations (Correlation Ratio)
    if len(continuous_columns) > 0 and len(categorical_columns) > 0:
        for cat_col in categorical_columns:
            for cont_col in continuous_columns:
                ratio = correlation_ratio(df[cat_col], df[cont_col])
                combined_corr.loc[cont_col, cat_col] = ratio
                combined_corr.loc[cat_col, cont_col] = ratio
    
    return combined_corr

def visualize_correlation_matrices(df):
    """
    Calculate and visualize a single correlation matrix for all columns.
    
    Args:
        df (pd.DataFrame): Input dataframe with mixed numerical and categorical columns
    """
    # Create combined correlation matrix
    combined_corr = create_combined_correlation_matrix(df)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        combined_corr,
        cmap='coolwarm',
        center=0,
        annot=True,
        fmt='.2f',
        square=True,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    
    plt.title('Combined Correlation Matrix\n(Pearson for Numerical, Theil\'s U for Categorical, Correlation Ratio for Mixed)')
    plt.tight_layout()
    plt.savefig('correlation_matrices.png')
    plt.close()

def main():
    # Example usage
    # Create a sample dataframe with mixed numerical and categorical columns
    np.random.seed(42)
    n_samples = 100
    
    # Create sample data
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'occupation': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Artist'], n_samples),
        'satisfaction': np.random.randint(1, 6, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Visualize correlation matrix
    visualize_correlation_matrices(df)

if __name__ == "__main__":
    main()