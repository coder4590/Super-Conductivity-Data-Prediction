import pandas as pd
import numpy as np
import csv
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
from sklearn.preprocessing import StandardScaler , LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split,KFold, cross_val_score, RandomizedSearchCV,TimeSeriesSplit
from scipy.stats import randint, uniform 
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo 
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from scipy.stats import gaussian_kde

superconductivty_data = fetch_ucirepo(id=464)

X = superconductivty_data.data.features 
Y = superconductivty_data.data.targets 
df = superconductivty_data.data.features.copy()


def analysis_data():
    df=X.copy()
    df['critical_temp'] = Y['critical_temp']


    print("Little detail of the like of hte data which are we working on")
    print(f"Columns name and type : {X.columns.tolist()}")
    print(f"column name of the target columns : {Y.columns.tolist()}")
    print(f"Data types of the columns : {X.dtypes}")
    print(f"Other type of detail : {X.describe()}")
    print(f"Length of the total data: {len(X)}")

    # This is the plot number One 
    ar_set1 = ['mean_atomic_radius', 'wtd_mean_atomic_radius', 'gmean_atomic_radius', 
           'wtd_gmean_atomic_radius', 'entropy_atomic_radius']
    ar_set2 = ['wtd_entropy_atomic_radius', 'range_atomic_radius', 'wtd_range_atomic_radius',
            'std_atomic_radius', 'wtd_std_atomic_radius']

    # Set 1 - Pivot Bars
    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    for idx, col in enumerate(ar_set1):
        df_temp = df[[col, 'critical_temp']].copy()
        df_temp['bin'] = pd.cut(df_temp[col], bins=10)
        pivot = df_temp.groupby('bin', observed=False)['critical_temp'].mean()
        
        axes[idx].bar(range(len(pivot)), pivot.values, color='steelblue', edgecolor='white')
        axes[idx].set_xticks(range(len(pivot)))
        axes[idx].set_xticklabels([f'{x.left:.0f}' for x in pivot.index], rotation=45, fontsize=7)
        axes[idx].set_xlabel(col.replace('_', ' ').title(), fontsize=9)
        axes[idx].set_ylabel('Avg Critical Temp (K)')
        axes[idx].grid(True, alpha=0.2)

    plt.suptitle('1. Atomic Radius (Set 1/2) — Avg Critical Temp per Bin', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot1_atomic_radius_pivot_set1.png', dpi=150)

    # Set 2 - Pivot Bars
    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    for idx, col in enumerate(ar_set2):
        df_temp = df[[col, 'critical_temp']].copy()
        df_temp['bin'] = pd.cut(df_temp[col], bins=10)
        pivot = df_temp.groupby('bin', observed=False)['critical_temp'].mean()
        
        axes[idx].bar(range(len(pivot)), pivot.values, color='steelblue', edgecolor='white')
        axes[idx].set_xticks(range(len(pivot)))
        axes[idx].set_xticklabels([f'{x.left:.0f}' for x in pivot.index], rotation=45, fontsize=7)
        axes[idx].set_xlabel(col.replace('_', ' ').title(), fontsize=9)
        axes[idx].set_ylabel('Avg Critical Temp (K)')
        axes[idx].grid(True, alpha=0.2)

    plt.suptitle('2. Atomic Radius (Set 2/2) — Avg Critical Temp per Bin', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot2_atomic_radius_pivot_set2.png', dpi=150) 

    # Plot number 2 which shows the like of the distribution of the FIE
    fie_set1 = ['mean_fie', 'wtd_mean_fie', 'gmean_fie', 'wtd_gmean_fie', 'entropy_fie']

    fig, axes = plt.subplots(1, 5, figsize=(25, 7))

    for idx, col in enumerate(fie_set1):
        ax = axes[idx]
        df_temp = df[[col, 'critical_temp']].copy()
        df_temp['bin'] = pd.cut(df_temp[col], bins=8)
        
        bins = df_temp['bin'].cat.categories
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(bins)))
        
        for i, b in enumerate(bins[::-1]):
            subset = df_temp[df_temp['bin'] == b]['critical_temp']
            if len(subset) > 10:
                kde = gaussian_kde(subset)
                xs = np.linspace(0, 200, 500)
                ys = kde(xs)
                ys = ys / ys.max() * 0.8
                ax.fill_between(xs, i, i + ys, alpha=0.7, color=colors[len(bins)-1-i])
                ax.plot(xs, i + ys, color='white', linewidth=0.5)
        
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        ax.set_xlim(0, 200)
        ax.set_ylim(-0.5, len(bins))
        ax.set_yticks(range(len(bins)))
        ax.set_yticklabels([f'{b.left:.0f}-{b.right:.0f}' for b in bins[::-1]], fontsize=7)
        ax.grid(True, alpha=0.2)

    plt.suptitle('2. FIE (Set 1/2) — Critical Temp Distribution by Feature Bins', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot3_fie_joy_set1.png', dpi=150)
    # Set Number 2
    fie_set2 = ['wtd_entropy_fie', 'range_fie', 'wtd_range_fie', 'std_fie', 'wtd_std_fie']

    fig, axes = plt.subplots(1, 5, figsize=(25, 7))

    for idx, col in enumerate(fie_set2):
        ax = axes[idx]
        df_temp = df[[col, 'critical_temp']].copy()
        df_temp['bin'] = pd.cut(df_temp[col], bins=8)
        
        bins = df_temp['bin'].cat.categories
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(bins)))
        
        for i, b in enumerate(bins[::-1]):
            subset = df_temp[df_temp['bin'] == b]['critical_temp']
            if len(subset) > 10:
                kde = gaussian_kde(subset)
                xs = np.linspace(0, 200, 500)
                ys = kde(xs)
                ys = ys / ys.max() * 0.8
                ax.fill_between(xs, i, i + ys, alpha=0.7, color=colors[len(bins)-1-i])
                ax.plot(xs, i + ys, color='white', linewidth=0.5)
        
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        ax.set_xlim(0, 200)
        ax.set_ylim(-0.5, len(bins))
        ax.set_yticks(range(len(bins)))
        ax.set_yticklabels([f'{b.left:.1f}-{b.right:.1f}' for b in bins[::-1]], fontsize=7)
        ax.grid(True, alpha=0.2)

    plt.suptitle('2. FIE (Set 2/2) — Critical Temp Distribution by Feature Bins', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot4_fie_joy_set2.png', dpi=150)

    # plot Number 3

    am_set1 = ['mean_atomic_mass', 'wtd_mean_atomic_mass', 'gmean_atomic_mass', 
            'wtd_gmean_atomic_mass', 'entropy_atomic_mass']

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    for idx, col in enumerate(am_set1):
        ax = axes[idx]
        df_temp = df[[col, 'critical_temp']].copy()
        df_temp['bin'] = pd.cut(df_temp[col], bins=10)
        
        grouped = df_temp.groupby('bin', observed=False)['critical_temp']
        means = grouped.mean()
        stds = grouped.std()
        centers = [b.mid for b in means.index]
        
        ax.errorbar(centers, means.values, yerr=stds.values, fmt='o-', 
                    color='steelblue', ecolor='gray', capsize=3, capthick=1, 
                    markersize=6, linewidth=2)
        ax.fill_between(centers, means.values - stds.values, 
                        means.values + stds.values, alpha=0.15, color='steelblue')
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Avg Critical Temp (K)')
        ax.grid(True, alpha=0.3)

    plt.suptitle('3. Atomic Mass (Set 1/2) — Mean Critical Temp with Std Error', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot5_atomic_mass_line_set1.png', dpi=150)

    # this is the set Number four 
    am_set2 = ['wtd_entropy_atomic_mass', 'range_atomic_mass', 'wtd_range_atomic_mass',
           'std_atomic_mass', 'wtd_std_atomic_mass']

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    for idx, col in enumerate(am_set2):
        ax = axes[idx]
        df_temp = df[[col, 'critical_temp']].copy()
        df_temp['bin'] = pd.cut(df_temp[col], bins=10)
        
        grouped = df_temp.groupby('bin', observed=False)['critical_temp']
        means = grouped.mean()
        stds = grouped.std()
        centers = [b.mid for b in means.index]
        
        ax.errorbar(centers, means.values, yerr=stds.values, fmt='o-', 
                    color='steelblue', ecolor='gray', capsize=3, capthick=1, 
                    markersize=6, linewidth=2)
        ax.fill_between(centers, means.values - stds.values, 
                        means.values + stds.values, alpha=0.15, color='steelblue')
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Avg Critical Temp (K)')
        ax.grid(True, alpha=0.3)

    plt.suptitle('3. Atomic Mass (Set 2/2) — Mean Critical Temp with Std Error', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot6_atomic_mass_line_set2.png', dpi=150)

    # this is the next plot 
    density_set1 = ['mean_Density', 'wtd_mean_Density', 'gmean_Density', 
                    'wtd_gmean_Density', 'entropy_Density']

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    for idx, col in enumerate(density_set1):
        ax = axes[idx]
        hb = ax.hexbin(df[col], df['critical_temp'], gridsize=30, cmap='YlOrRd',
                    mincnt=1, bins='log')
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Critical Temp (K)')
        plt.colorbar(hb, ax=ax, label='Count (log scale)')

    plt.suptitle('4. Density (Set 1/2) — Hexbin: Critical Temp vs Feature', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot7_density_hexbin_set1.png', dpi=150)
    # Density - Set 2
    density_set2 = ['wtd_entropy_Density', 'range_Density', 'wtd_range_Density',
                    'std_Density', 'wtd_std_Density']

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    for idx, col in enumerate(density_set2):
        ax = axes[idx]
        hb = ax.hexbin(df[col], df['critical_temp'], gridsize=30, cmap='YlOrRd',
                    mincnt=1, bins='log')
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Critical Temp (K)')
        plt.colorbar(hb, ax=ax, label='Count (log scale)')

    plt.suptitle('4. Density (Set 2/2) — Hexbin: Critical Temp vs Feature', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot8_density_hexbin_set2.png', dpi=150)

    # Electron Affinity - Set 1
    ea_set1 = ['mean_ElectronAffinity', 'wtd_mean_ElectronAffinity', 'gmean_ElectronAffinity',
            'wtd_gmean_ElectronAffinity', 'entropy_ElectronAffinity']

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    for idx, col in enumerate(ea_set1):
        ax = axes[idx]
        df_temp = df[[col, 'critical_temp']].copy()
        df_temp['bin'] = pd.cut(df_temp[col], bins=10)
        
        grouped = df_temp.groupby('bin', observed=False)['critical_temp']
        means = grouped.mean()
        stds = grouped.std()
        centers = [b.mid for b in means.index]
        
        ax.errorbar(centers, means.values, yerr=stds.values, fmt='o-',
                    color='steelblue', ecolor='gray', capsize=3, capthick=1,
                    markersize=6, linewidth=2)
        ax.fill_between(centers, means.values - stds.values,
                        means.values + stds.values, alpha=0.15, color='steelblue')
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Avg Critical Temp (K)')
        ax.grid(True, alpha=0.3)

    plt.suptitle('5. Electron Affinity (Set 1/2) — Mean Critical Temp with Std Error', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot9_electron_affinity_line_set1.png', dpi=150)
    
    # Electron Affinity - Set 2
    ea_set2 = ['wtd_entropy_ElectronAffinity', 'range_ElectronAffinity', 'wtd_range_ElectronAffinity',
            'std_ElectronAffinity', 'wtd_std_ElectronAffinity']

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    for idx, col in enumerate(ea_set2):
        ax = axes[idx]
        df_temp = df[[col, 'critical_temp']].copy()
        df_temp['bin'] = pd.cut(df_temp[col], bins=10)
        
        grouped = df_temp.groupby('bin', observed=False)['critical_temp']
        means = grouped.mean()
        stds = grouped.std()
        centers = [b.mid for b in means.index]
        
        ax.errorbar(centers, means.values, yerr=stds.values, fmt='o-',
                    color='steelblue', ecolor='gray', capsize=3, capthick=1,
                    markersize=6, linewidth=2)
        ax.fill_between(centers, means.values - stds.values,
                        means.values + stds.values, alpha=0.15, color='steelblue')
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Avg Critical Temp (K)')
        ax.grid(True, alpha=0.3)

    plt.suptitle('5. Electron Affinity (Set 2/2) — Mean Critical Temp with Std Error', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot10_electron_affinity_line_set2.png', dpi=150)
    
    # fusion heat set Number 1
    fh_set1 = ['mean_FusionHeat', 'wtd_mean_FusionHeat', 'gmean_FusionHeat', 
           'wtd_gmean_FusionHeat', 'entropy_FusionHeat']

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 5))

    for idx, col in enumerate(fh_set1):
        ax = axes[idx]
        df_temp = df[[col, 'critical_temp']].copy()
        df_temp['bin'] = pd.cut(df_temp[col], bins=8)
        
        grouped = df_temp.groupby('bin', observed=False)['critical_temp']
        means = grouped.mean()
        stds = grouped.std()
        labels = [f'{b.left:.0f}' for b in means.index]
        
        bars = ax.bar(range(len(means)), means.values, yerr=stds.values, 
                    color=colors[idx], edgecolor='white', capsize=3, alpha=0.85)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Avg Critical Temp (K)')
        ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle('6. Fusion Heat (Set 1/2) — Bar Chart with Std Error', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot11_fusion_heat_bar_set1.png', dpi=150)
    
    # Fusionheat set Number 2
    fh_set2 = ['wtd_entropy_FusionHeat', 'range_FusionHeat', 'wtd_range_FusionHeat',
           'std_FusionHeat', 'wtd_std_FusionHeat']

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, 5))

    for idx, col in enumerate(fh_set2):
        ax = axes[idx]
        df_temp = df[[col, 'critical_temp']].copy()
        df_temp['bin'] = pd.cut(df_temp[col], bins=8)
        
        grouped = df_temp.groupby('bin', observed=False)['critical_temp']
        means = grouped.mean()
        stds = grouped.std()
        labels = [f'{b.left:.0f}' for b in means.index]
        
        ax.bar(range(len(means)), means.values, yerr=stds.values,
            color=colors[idx], edgecolor='white', capsize=3, alpha=0.85)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Avg Critical Temp (K)')
        ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle('6. Fusion Heat (Set 2/2) — Bar Chart with Std Error', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot12_fusion_heat_bar_set2.png', dpi=150)
    
    # thermal conductivity Set Number 1

    tc_set1 = ['mean_ThermalConductivity', 'wtd_mean_ThermalConductivity', 'gmean_ThermalConductivity',
           'wtd_gmean_ThermalConductivity', 'entropy_ThermalConductivity']

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
    colors = plt.cm.cividis(np.linspace(0.2, 0.8, 5))

    for idx, col in enumerate(tc_set1):
        ax = axes[idx]
        df_temp = df[[col, 'critical_temp']].copy()
        df_temp['bin'] = pd.cut(df_temp[col], bins=8)
        
        grouped = df_temp.groupby('bin', observed=False)['critical_temp']
        means = grouped.mean()
        stds = grouped.std()
        labels = [f'{b.left:.0f}' for b in means.index]
        
        ax.bar(range(len(means)), means.values, yerr=stds.values,
            color=colors[idx], edgecolor='white', capsize=3, alpha=0.85)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Avg Critical Temp (K)')
        ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle('7. Thermal Conductivity (Set 1/2) — Bar Chart with Std Error', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot13_thermal_conductivity_bar_set1.png', dpi=150)
    
    #Fusion Heat Set Number 2
    tc_set2 = ['wtd_entropy_ThermalConductivity', 'range_ThermalConductivity', 'wtd_range_ThermalConductivity',
           'std_ThermalConductivity', 'wtd_std_ThermalConductivity']

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
    colors = plt.cm.cividis(np.linspace(0.2, 0.8, 5))

    for idx, col in enumerate(tc_set2):
        ax = axes[idx]
        df_temp = df[[col, 'critical_temp']].copy()
        df_temp['bin'] = pd.cut(df_temp[col], bins=8)
        
        grouped = df_temp.groupby('bin', observed=False)['critical_temp']
        means = grouped.mean()
        stds = grouped.std()
        labels = [f'{b.left:.0f}' for b in means.index]
        
        ax.bar(range(len(means)), means.values, yerr=stds.values,
            color=colors[idx], edgecolor='white', capsize=3, alpha=0.85)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Avg Critical Temp (K)')
        ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle('7. Thermal Conductivity (Set 2/2) — Bar Chart with Std Error', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot14_thermal_conductivity_bar_set2.png', dpi=150)
    
    # Valence shell Set Number One
    # Valence - Set 1
    val_set1 = ['mean_Valence', 'wtd_mean_Valence', 'gmean_Valence',
                'wtd_gmean_Valence', 'entropy_Valence']

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    for idx, col in enumerate(val_set1):
        ax = axes[idx]
        df_temp = df[[col, 'critical_temp']].copy()
        df_temp['bin'] = pd.cut(df_temp[col], bins=10)
        
        grouped = df_temp.groupby('bin', observed=False)['critical_temp']
        means = grouped.mean()
        stds = grouped.std()
        centers = [b.mid for b in means.index]
        
        ax.errorbar(centers, means.values, yerr=stds.values, fmt='o-',
                    color='steelblue', ecolor='gray', capsize=3, capthick=1,
                    markersize=6, linewidth=2)
        ax.fill_between(centers, means.values - stds.values,
                        means.values + stds.values, alpha=0.15, color='steelblue')
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Avg Critical Temp (K)')
        ax.grid(True, alpha=0.3)

    plt.suptitle('8. Valence (Set 1/2) — Mean Critical Temp with Std Error', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot15_valence_line_set1.png', dpi=150)
    
    # Valence Shell Set Number 2

    val_set2 = ['wtd_entropy_Valence', 'range_Valence', 'wtd_range_Valence',
            'std_Valence', 'wtd_std_Valence']

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    for idx, col in enumerate(val_set2):
        ax = axes[idx]
        df_temp = df[[col, 'critical_temp']].copy()
        df_temp['bin'] = pd.cut(df_temp[col], bins=10)
        
        grouped = df_temp.groupby('bin', observed=False)['critical_temp']
        means = grouped.mean()
        stds = grouped.std()
        centers = [b.mid for b in means.index]
        
        ax.errorbar(centers, means.values, yerr=stds.values, fmt='o-',
                    color='steelblue', ecolor='gray', capsize=3, capthick=1,
                    markersize=6, linewidth=2)
        ax.fill_between(centers, means.values - stds.values,
                        means.values + stds.values, alpha=0.15, color='steelblue')
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Avg Critical Temp (K)')
        ax.grid(True, alpha=0.3)

    plt.suptitle('8. Valence (Set 2/2) — Mean Critical Temp with Std Error', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot16_valence_line_set2.png', dpi=150)
    plt.show()
