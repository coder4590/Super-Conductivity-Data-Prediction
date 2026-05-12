def clean_data(X,Y):
    df=X.copy()
    df['critical_temp'] = Y['critical_temp']

    print("Missing value check ===========================================")

    print("To see if how many of the values are missing ")
    print(f"Missing Values: {X.isnull().sum()}")
    "There is not a single missing values in this data set and the like of it"

    print("Duplicate Value check ==========================================")

    print("to check the duplicate values")
    print(f"Duplicate Values Check : {X.duplicated().sum()}")
    "this data set contain some of the missing values so we are dropping all of those missing values from this datset"
    df=df.drop_duplicates()

    print("Adding Data Quality Checks ======================================")
    df = df[df['critical_temp'] >= 0]
    df = df[df['number_of_elements'] >= 1]
    df = df[df['number_of_elements'] <= 10]

    print("Adding outlier detection ========================================")

    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    n_features = len(numerical_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(numerical_features):
        axes[idx].boxplot(df[col])
        axes[idx].set_title(f'{col} - Outlier Check')
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        print(f"{col}: {len(outliers)} outliers detected")

    # Hide unused subplots
    for idx in range(len(numerical_features), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig("Outlier_checker.png")

    print("drop Columns ==========================================")

    drop_cols = [
    'wtd_mean_atomic_mass','gmean_atomic_mass','wtd_gmean_atomic_mass','wtd_entropy_atomic_mass','range_atomic_mass','wtd_range_atomic_mass','wtd_std_atomic_mass',
    'wtd_mean_fie','gmean_fie','wtd_gmean_fie','range_fie','wtd_range_fie','wtd_std_fie',
    'wtd_mean_atomic_radius','gmean_atomic_radius','wtd_gmean_atomic_radius','wtd_entropy_atomic_radius','wtd_range_atomic_radius','wtd_std_atomic_radius',
    'mean_Density','wtd_mean_Density','gmean_Density','wtd_gmean_Density','range_Density','wtd_range_Density','wtd_std_Density',
    'wtd_mean_ElectronAffinity','gmean_ElectronAffinity','wtd_gmean_ElectronAffinity','wtd_entropy_ElectronAffinity','wtd_range_ElectronAffinity','wtd_std_ElectronAffinity',
    'wtd_mean_FusionHeat','gmean_FusionHeat','wtd_gmean_FusionHeat','wtd_entropy_FusionHeat','wtd_range_FusionHeat','wtd_std_FusionHeat',
    'wtd_mean_ThermalConductivity','gmean_ThermalConductivity','wtd_gmean_ThermalConductivity','wtd_entropy_ThermalConductivity','wtd_range_ThermalConductivity','wtd_std_ThermalConductivity',
    'wtd_mean_Valence','gmean_Valence','wtd_gmean_Valence'
]
    df.drop(columns=drop_cols, inplace=True)

    return df
