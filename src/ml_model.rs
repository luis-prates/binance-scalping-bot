// ml_model.rs
use crate::strategy::MarketData;
use anyhow::Result;
use log::{info, warn};
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use smartcore::{
    ensemble::random_forest_regressor::RandomForestRegressor,
    linalg::naive::dense_matrix::DenseMatrix, model_selection::train_test_split,
};
use std::fs;
use std::path::Path;

#[derive(Debug)]
pub struct MLPredictor {
    model: Option<RandomForestRegressor<f64>>,
    feature_window: usize,
    prediction_threshold: f64,
    training_data: Vec<(Vec<f64>, f64)>,
}

impl MLPredictor {
    pub fn new() -> Self {
        Self {
            model: None,
            feature_window: 20, // Look back window for features
            prediction_threshold: 0.6,
            training_data: Vec::new(),
        }
    }

    pub fn predict(&self, features: Vec<f64>) -> Result<f64> {
        if features.is_empty() {
            return Ok(0.5); // Default neutral prediction
        }

        if let Some(ref model) = self.model {
            let features_matrix = DenseMatrix::from_vec(1, features.len(), &features);
            let prediction = model.predict(&features_matrix)?;
            Ok(prediction[0])
        } else {
            warn!("No trained model available, returning neutral prediction");
            Ok(0.5)
        }
    }

    pub fn prepare_features(&self, market_data: &MarketData) -> Vec<f64> {
        let mut features = Vec::new();

        // Price-based features
        if let Some(latest_price) = market_data.get_latest_price() {
            // Normalize price using the mean of last N prices
            let mean_price: Decimal = market_data.prices.iter().sum::<Decimal>()
                / Decimal::from(market_data.prices.len());

            features.push((latest_price / mean_price).to_f64().unwrap_or(1.0));
        }

        // Volume features
        if let Some(latest_volume) = market_data.volumes.back() {
            let mean_volume: Decimal = market_data.volumes.iter().sum::<Decimal>()
                / Decimal::from(market_data.volumes.len());

            features.push((latest_volume / mean_volume).to_f64().unwrap_or(1.0));
        }

        // Price changes (returns)
        if market_data.prices.len() >= 2 {
            let price_changes: Vec<f64> = market_data
                .prices
                .iter()
                .zip(market_data.prices.iter().skip(1))
                .map(|(prev, curr)| ((curr - prev) / prev).to_f64().unwrap_or(0.0))
                .take(5) // Use last 5 price changes
                .collect();

            features.extend(price_changes);
        }

        // Spread feature
        if let Some(spread) = market_data.get_spread() {
            features.push(spread.to_f64().unwrap_or(0.0));
        }

        // Time-based features
        if let Some(latest_timestamp) = market_data.timestamps.back() {
            // Hour of day (normalized)
            let hour = (latest_timestamp % 86400) / 3600;
            features.push(hour as f64 / 24.0);
        }

        features
    }

    pub fn train(&mut self, market_data: &MarketData, labels: &[f64]) -> Result<()> {
        if market_data.prices.len() - 1 != labels.len() {
            warn!(
                "Data and labels length mismatch: {} vs {}",
                market_data.prices.len(),
                labels.len()
            );
            return Err(anyhow::anyhow!("Data and labels length mismatch"));
        }

        let mut training_features = Vec::new();
        let mut training_labels = Vec::new();

        // Prepare training data
        for i in self.feature_window..market_data.prices.len() - 1 {
            let window_data = MarketData {
                prices: market_data
                    .prices
                    .range(i - self.feature_window..i)
                    .cloned()
                    .collect(),
                volumes: market_data
                    .volumes
                    .range(i - self.feature_window..i)
                    .cloned()
                    .collect(),
                timestamps: market_data
                    .timestamps
                    .range(i - self.feature_window..i)
                    .cloned()
                    .collect(),
                orderbook: market_data.orderbook.clone(),
                ticker_24hr: market_data.ticker_24hr.clone(),
            };

            let features = self.prepare_features(&window_data);
            training_features.push(features);
            training_labels.push(labels[i]);
        }

        // Convert to matrix format
        let x = DenseMatrix::from_2d_vec(&training_features);
        let y = training_labels;

        // Split data
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);

        // Train model
        let model = RandomForestRegressor::fit(&x_train, &y_train, Default::default())?;

        // Evaluate model
        let predictions = model.predict(&x_test)?;
        let mse = predictions
            .iter()
            .zip(y_test.iter())
            .map(|(pred, actual)| (pred - actual).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;

        info!("Model trained with MSE: {:.6}", mse);

        self.model = Some(model);
        Ok(())
    }

    pub fn save_model(&self, path: &Path) -> Result<()> {
        if let Some(ref model) = self.model {
            let serialized = serde_json::to_string(&model)?;
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            info!("Saving model to {}", path.display());
            fs::write(path, serialized)?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("No model to save"))
        }
    }

    pub fn load_model(&mut self, path: &Path) -> Result<()> {
        let contents = fs::read_to_string(path)?;
        self.model = Some(serde_json::from_str(&contents)?);
        Ok(())
    }

    pub fn add_training_data(&mut self, features: Vec<f64>, label: f64) {
        self.training_data.push((features, label));
    }

    pub fn retrain(&mut self) -> Result<()> {
        if self.training_data.is_empty() {
            return Ok(());
        }

        let (features, labels): (Vec<_>, Vec<_>) = self.training_data.iter().cloned().unzip();
        let x = DenseMatrix::from_2d_vec(&features);

        // Train new model
        let model = RandomForestRegressor::fit(&x, &labels, Default::default())?;

        self.model = Some(model);
        Ok(())
    }
}
