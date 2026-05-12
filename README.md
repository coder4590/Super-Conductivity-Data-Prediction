# ❄️ Predicting Superconductivity — The Temperature Where Magic Happens

> Some materials, when cooled enough, become perfect conductors. Zero electrical resistance. Current flows forever. Levitating trains. Lossless power grids. MRI machines that save lives.

> The catch? Most materials only superconduct at temperatures colder than deep space. Finding materials that work at higher temperatures is like finding a needle in a haystack — except the haystack is every possible combination of elements in the periodic table.

> What if machine learning could scan thousands of known compounds and learn the patterns? What if it could tell scientists: *"Here. Look here. This combination of atomic properties works."*

> **That's what this project does.**

---

## 🔬 What This Actually Is

A machine learning regression system that predicts the critical temperature at which a material becomes superconducting, using only the atomic properties of its constituent elements. No lab experiments. No expensive measurements. Just physics-inspired features and a trained model that learned what makes a good superconductor.

| Input | Output |
|---|---|
| 33 atomic property features | Critical Temperature (Kelvin) |
| 21,263 chemical compounds | 0.948 R² score |
| Random split with cross-validation | Honest evaluation, no leakage |

---

## 📊 The Data — Pre-Engineered Physics Features

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Records | 21,263 chemical compounds |
| Original Features | 81 statistical summaries of atomic properties |
| Kept Features | 33 (48 dropped after EDA) |
| Target | Critical temperature (0 to 185 Kelvin) |
| Data Type | Cross-sectional. Random split. No time dependency. |

---

## 💡 The Core Insight — 81 Features, 8 Properties, Mostly Redundant

This dataset is unique. The original creators computed every possible statistical summary for 8 atomic properties — mean, weighted mean, geometric mean, weighted geometric mean, entropy, weighted entropy, range, weighted range, standard deviation, weighted standard deviation. 8 properties × 10 statistics each = 80 features plus number of elements. 81 total. But here's the problem: most of these statistics carry the same information. Mean and weighted mean are nearly identical. Standard deviation and range often show the same pattern. Keeping all 81 features doesn't help the model — it floods it with noise and redundancy.

---

## 🔍 EDA — Finding Physics in the Numbers

Every atomic property was analyzed in two sets of five features each. Four different plot types were used across all properties. Each plot was designed to answer one question: does this statistic capture something unique about superconductivity, or is it just another version of the mean?

---

### Property 1: Atomic Radius
<img width="3750" height="900" alt="image" src="https://github.com/user-attachments/assets/57623e28-dc83-4838-8352-10dc1fd7e94c" />

<img width="3750" height="900" alt="image" src="https://github.com/user-attachments/assets/cb363882-bfe3-4843-a87b-97695c203b62" />

| Plot Type | Scatter Plot (5 per row) |
|---|---|
| **Set 1 Features** | Mean, Weighted Mean, Geometric Mean, Weighted Geometric Mean, Entropy |
| **Finding** | Mean peaked at mid-values — there's a Goldilocks zone for atomic radius. All four mean variants showed identical patterns. Entropy was flat. |
| **Decision** | Kept mean_atomic_radius. Dropped all 4 redundant variants. |
| **Set 2 Features** | Weighted Entropy, Range, Weighted Range, Std, Weighted Std |
| **Finding** | Range and Std show clear peaks at mid-values but similar to each other. Entropy flat. |
| **Decision** | Kept range_atomic_radius and std_atomic_radius. Dropped 3. |
| **Physics Insight** | Atomic radius determines the crystal lattice structure. A specific size creates optimal vibrational modes for electron pairing. Too small or too large kills superconductivity. |

---

### Property 2: First Ionization Energy (FIE)
<img width="3750" height="1050" alt="image" src="https://github.com/user-attachments/assets/192c8720-cce3-4e55-9a24-801b2b2ab8d2" />

<img width="3750" height="1050" alt="image" src="https://github.com/user-attachments/assets/874b784f-70e3-407c-b14e-eefb998c8114" />

| Plot Type | Joy Plot / Ridge Plot |
|---|---|
| **Set 1 Features** | Mean, Weighted Mean, Geometric Mean, Weighted Geometric Mean, Entropy |
| **Finding** | Higher ionization energy shifts the critical temperature distribution rightward. The weighted entropy shows a distinct mountain pattern that the means don't capture. |
| **Decision** | Kept mean_fie, entropy_fie. Dropped 3 redundant means. |
| **Set 2 Features** | Weighted Entropy, Range, Weighted Range, Std, Weighted Std |
| **Finding** | Weighted entropy and std show genuine signal. Range is weak. |
| **Decision** | Kept wtd_entropy_fie, std_fie. Dropped 3. |
| **Physics Insight** | Ionization energy correlates with bond strength. Higher FIE means tighter electron bonds and higher-frequency lattice vibrations — a key ingredient for higher critical temperatures. |

---

### Property 3: Atomic Mass
<img width="3750" height="900" alt="image" src="https://github.com/user-attachments/assets/e8f386bb-d4d2-4849-aab6-34fdd8f5d236" />
<img width="3750" height="900" alt="image" src="https://github.com/user-attachments/assets/b3964f67-5be9-4ae9-88b0-8e23efc9636a" />

| Plot Type | Line Plot with Error Bars |
|---|---|
| **Set 1 Features** | Mean, Weighted Mean, Geometric Mean, Weighted Geometric Mean, Entropy |
| **Finding** | Mean peaks at mid-range (40-60 atomic mass). All mean variants identical. Entropy shows a different shape — worth keeping. |
| **Decision** | Kept mean_atomic_mass, entropy_atomic_mass. Dropped 3. |
| **Set 2 Features** | Weighted Entropy, Range, Weighted Range, Std, Weighted Std |
| **Finding** | Std shows a clear peak. Weighted range has a distinct pattern. Range and weighted entropy are weak. |
| **Decision** | Kept std_atomic_mass, wtd_range_atomic_mass. Dropped 3. |
| **Physics Insight** | Atomic mass affects phonon frequencies. The sweet spot at mid-range mass creates optimal lattice dynamics. Std captures variance — different masses in the compound create anharmonic potentials that enhance electron pairing. |

---

### Property 4: Density
<img width="3750" height="900" alt="image" src="https://github.com/user-attachments/assets/c60aef6a-bf0c-4ced-a268-59446301007e" />
<img width="3750" height="900" alt="image" src="https://github.com/user-attachments/assets/9338c2d3-9c7f-442e-a995-d69b512dfe37" />

| Plot Type | Hexbin Plot |
|---|---|
| **Set 1 Features** | Mean, Weighted Mean, Geometric Mean, Weighted Geometric Mean, Entropy |
| **Finding** | Most compounds cluster at low density and low critical temp. Means are identical. Entropy shows a different spread pattern. |
| **Decision** | Kept entropy_Density. Dropped 4 redundant means. |
| **Set 2 Features** | Weighted Entropy, Range, Weighted Range, Std, Weighted Std |
| **Finding** | Std and weighted entropy show structure not seen in Set 1. Range is weak. |
| **Decision** | Kept wtd_entropy_Density, std_Density. Dropped 3. |
| **Physics Insight** | Density is a bulk property — less directly linked to superconductivity than atomic-level features. But its variance captures structural heterogeneity that matters. |

---

### Property 5: Electron Affinity
<img width="3750" height="900" alt="image" src="https://github.com/user-attachments/assets/6d4d4ada-2afe-4576-b294-ffa555050b32" />

<img width="3750" height="900" alt="image" src="https://github.com/user-attachments/assets/b3349ab1-a8c0-4a2a-ad46-6bfed0114aea" />

| Plot Type | Line Plot with Error Bars |
|---|---|
| **Set 1 Features** | Mean, Weighted Mean, Geometric Mean, Weighted Geometric Mean, Entropy |
| **Finding** | Mean and entropy show distinct patterns. Weighted gmean shows borderline signal. Other means are redundant. |
| **Decision** | Kept mean_ElectronAffinity, entropy_ElectronAffinity, wtd_gmean_ElectronAffinity. Dropped 2. |
| **Set 2 Features** | Weighted Entropy, Range, Weighted Range, Std, Weighted Std |
| **Finding** | Range electron affinity is the strongest signal of all — none of the others match it. Std also distinct. |
| **Decision** | Kept range_ElectronAffinity, std_ElectronAffinity. Dropped 3. |
| **Physics Insight** | Electron affinity determines charge transfer between atoms. The range captures chemical heterogeneity — a large difference creates charge doping that directly enables superconductivity. This was the strongest single-property finding. |

---

### Property 6: Fusion Heat
<img width="3750" height="900" alt="image" src="https://github.com/user-attachments/assets/7375ea98-ab53-4b84-9401-ba9b651bd0b4" />

<img width="3750" height="900" alt="image" src="https://github.com/user-attachments/assets/41245c12-e3c3-4191-b990-7150d7a8b7d6" />

| Plot Type | Bar Chart with Error Bars |
|---|---|
| **Set 1 Features** | Mean, Weighted Mean, Geometric Mean, Weighted Geometric Mean, Entropy |
| **Finding** | Entropy is the standout — all means are identical. Clear separation. |
| **Decision** | Kept entropy_FusionHeat, mean_FusionHeat. Dropped 3. |
| **Set 2 Features** | Weighted Entropy, Range, Weighted Range, Std, Weighted Std |
| **Finding** | Range is clean. Std is distinct. Others are redundant. |
| **Decision** | Kept range_FusionHeat, std_FusionHeat. Dropped 3. |
| **Physics Insight** | Fusion heat measures bond-breaking energy. Its entropy captures the diversity of bond strengths — a mix of strong and weak bonds creates anharmonic lattice vibrations that enhance superconductivity. |

---

### Property 7: Thermal Conductivity
<img width="3750" height="900" alt="image" src="https://github.com/user-attachments/assets/b32a2e56-77f7-4bb3-8954-7ee59bc41a20" />

<img width="3750" height="900" alt="image" src="https://github.com/user-attachments/assets/8affdb93-8862-4414-b797-88cb22351f60" />

| Plot Type | Bar Chart with Error Bars |
|---|---|
| **Set 1 Features** | Mean, Weighted Mean, Geometric Mean, Weighted Geometric Mean, Entropy |
| **Finding** | Mean peaks at mid-range. Entropy rises with higher values — opposite pattern. Both unique. |
| **Decision** | Kept mean_ThermalConductivity, entropy_ThermalConductivity. Dropped 3. |
| **Set 2 Features** | Weighted Entropy, Range, Weighted Range, Std, Weighted Std |
| **Finding** | Range is the only keeper. All others are redundant with Set 1 or each other. |
| **Decision** | Kept range_ThermalConductivity. Dropped 4. |
| **Physics Insight** | Thermal conductivity controls heat flow through the lattice. A wide range means some atoms conduct heat well while others don't — creating thermal gradients that enhance electron-phonon coupling. |

---

### Property 8: Valence Electrons
<img width="3750" height="900" alt="image" src="https://github.com/user-attachments/assets/ceb20fa6-ff49-4b66-a865-87afdd840901" />
<img width="3750" height="900" alt="image" src="https://github.com/user-attachments/assets/ba4c5fed-7f51-4fa2-a00f-ca68f027afdc" />

| Plot Type | Line Plot with Error Bars |
|---|---|
| **Set 1 Features** | Mean, Weighted Mean, Geometric Mean, Weighted Geometric Mean, Entropy |
| **Finding** | Mean and entropy are both strong. The mean variants are redundant. |
| **Decision** | Kept mean_Valence, entropy_Valence. Dropped 3. |
| **Set 2 Features** | Weighted Entropy, Range, Weighted Range, Std, Weighted Std |
| **Finding** | Every single feature in Set 2 shows a unique, distinct pattern. None are identical to each other. None are identical to Set 1. Valence is the most important property. |
| **Decision** | Kept all 5. Dropped none from Set 2. |
| **Physics Insight** | Valence electrons are the outermost electrons — the ones that form bonds and carry current. Every statistical view captures a different aspect of this critical property. This is why 7 of 10 valence features survived — more than any other property. |

---

## 📊 EDA Summary — 81 → 33 Features

| Property | Kept | Dropped |
|---|---|---|
| Atomic Radius | 3 | 7 |
| FIE | 4 | 6 |
| Atomic Mass | 4 | 6 |
| Density | 3 | 7 |
| Electron Affinity | 5 | 5 |
| Fusion Heat | 4 | 6 |
| Thermal Conductivity | 3 | 7 |
| Valence | 7 | 3 |
| **Total** | **33** | **48** |

---

## 🧪 The Model — Regression with Real Physics

| Metric | Value |
|---|---|
| Algorithm | XGBoost Regressor |
| Cross-Validation | 5-fold KFold |
| CV R² Score | 0.920 |
| Test R² Score | 0.948 |
| Mean Absolute Error | 4.33 Kelvin |
| Features Used | 33 (after dropping 48 redundant features) |

---

## 📈 Why 0.948 is Honest

| Check | Result |
|---|---|
| CV R² vs Test R² gap | 0.920 vs 0.948 — no overfitting. Test is actually higher. |
| Leakage check | No time series. No target in features. Clean random split. |
| Feature selection | 48 redundant features dropped through EDA. Every kept feature has physical justification. |
| Domain alignment | Valence (7 features kept) is the dominant property — matches known physics. |

This isn't a Kaggle hack. It's what happens when you remove noise from physics data and let the model learn from real signal.

---

## 🏗️ Pipeline Architecture

| Stage | What Happened |
|---|---|
| **Data Cleaning** | Missing values checked. Duplicates removed. Impossible values filtered. Outlier detection with boxplot grid. |
| **EDA** | 8 properties analyzed. 4 plot types. 16 total plots. Feature selection justified with physics. |
| **Feature Selection** | 81 → 33 features. Redundant statistics dropped. Unique signals preserved. |
| **Train/Test Split** | Random 80/20 split. No time dependency. Stratification not needed for regression. |
| **Preprocessing** | StandardScaler for all numeric features. |
| **Model** | XGBoost Regressor with 30 iterations of RandomizedSearchCV across 9 hyperparameters. |
| **Evaluation** | R² score. Mean Absolute Error. CV vs test comparison. |

---

## 📈 Key Metrics

| Metric | Value | Meaning |
|---|---|---|
| Test R² | 0.948 | 94.8% of variance explained |
| CV R² | 0.920 | Consistent across folds — no overfitting |
| MAE | 4.33 K | Average prediction off by 4.33 degrees Kelvin |
| Features from 81 | 33 | 48 redundant features removed |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn (Scatter, Joy, Line+Error, Hexbin, Bar) |
| ML Model | XGBoost Regressor |
| Preprocessing | StandardScaler |
| Tuning | RandomizedSearchCV, KFold |
| Evaluation | r2_score, mean_absolute_error |

---

## 📝 Lessons Learned

| # | Lesson |
|---|---|
| 1 | Pre-engineered features need EDA as much as raw data. The dataset creator computed everything — your job is to decide what matters. |
| 2 | Redundancy is the enemy. 48 features carried duplicate information. Removing them didn't hurt the model — it helped it. |
| 3 | Physics is predictable. Unlike human behavior or industrial sensors, atomic properties follow mathematical laws. High R² scores are normal when features align with domain knowledge. |
| 4 | Valence electrons are king. 7 of 10 valence features survived. Every statistical view added new information. The physics confirms what the EDA found. |
| 5 | Know when to celebrate. 0.948 on a physics dataset with clean feature selection is genuine. Not leakage. Not overfitting. Just good engineering. |

---

## 👤 Author

A machine learning engineer who believes that the next room-temperature superconductor might be hiding in a dataset — and someone needs to find it.

---

*"The periodic table has spoken. The patterns are there. Someone just needed to look."*
