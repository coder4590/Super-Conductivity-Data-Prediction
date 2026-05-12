def predict_data():
    df=clean_data(X,Y)
    target= 'critical_temp'

    removed_feature=[target]
    numerical_features=df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object','category']).columns.tolist()


    final_numerical_col=[col for col in numerical_features if col not in removed_feature]
    final_categorical_col=[col for col in categorical_features if col not in removed_feature]

    x=df[final_numerical_col+ final_categorical_col]
    y=df[target]

    x_train,x_test,y_train,y_test=train_test_split(
        x,y,
        test_size=42,
    )

    preprocessor=ColumnTransformer(
        [
            ('num', StandardScaler(), final_numerical_col),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), final_categorical_col)
        ]
    )

    pipelines=Pipeline(
        [
            ('pre', preprocessor),
            ('model',xgb.XGBRegressor(random_state=42))
        ]
    )

    param_dist = {
    'model__n_estimators': randint(200, 600),
    'model__learning_rate': uniform(0.01, 0.3),
    'model__max_depth': randint(4, 12),
    'model__subsample': uniform(0.6, 0.4),
    'model__colsample_bytree': uniform(0.6, 0.4),
    'model__reg_lambda': uniform(0, 10),
    'model__reg_alpha': uniform(0, 5),
    'model__gamma': uniform(0, 5),
    'model__min_child_weight': randint(1, 10)
}
    search = RandomizedSearchCV(
        estimator=pipelines,
        param_distributions=param_dist,
        n_iter=30,  # Increase if you have time
        cv=5, # using the cross validation with the 5 cross cv 
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    search.fit(x_train,y_train)

    print("Best Parameters:", search.best_params_)
    print("Best CV R2:", search.best_score_)


    y_pred=search.best_estimator_.predict(x_test)

    mae=mean_absolute_error(y_test, y_pred)
    r2=r2_score(y_test, y_pred)

    print("\n MODEL PERFORMANCE:")
    print(f"   r² Score: {r2:.3f} (1.0 = perfect prediction)")
    print(f"   Mean Absolute Error: {mae:,.2f}")
    print(f"   Average prediction is off by: {mae:,.2f}")
    








if __name__ == "__main__":
    predict_data()
