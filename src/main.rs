use csv::ReaderBuilder;
use linfa::prelude::*;
use linfa_svm::SvmParams;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayBase;
use ndarray::Axis;
use ndarray::Dim;
use ndarray::OwnedRepr;
use rand::seq::SliceRandom;
use reqwest::blocking::get;
use std::collections::HashMap;
use std::error::Error;
use std::io::Cursor;

// Function to download Titanic dataset
fn download_titanic_data() -> Result<String, Box<dyn Error>> {
    let url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv";
    let response = get(url)?.text()?;
    Ok(response)
}

// Function to load Titanic data into ndarray
fn load_titanic_to_ndarray(data: &str) -> Result<(Vec<String>, Array2<f64>), Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().from_reader(Cursor::new(data));

    // Get the headers (metadata)
    let headers: Vec<String> = reader.headers()?.iter().map(|s| s.to_string()).collect();

    // Store column indices for categorical columns
    let categorical_columns = vec!["Sex", "Embarked"]; // Example categorical columns
    let mut cat_col_indices: HashMap<String, usize> = HashMap::new();
    for (i, header) in headers.iter().enumerate() {
        if categorical_columns.contains(&header.as_str()) {
            cat_col_indices.insert(header.clone(), i);
        }
    }

    // Extract records and convert to f64 ndarray with one-hot encoding for categorical columns
    let mut records = Vec::new();
    let mut one_hot_data = Vec::new();
    for result in reader.records() {
        let record = result?;
        let mut row = vec![0.0; headers.len()];

        for (i, field) in record.iter().enumerate() {
            if let Some(cat_index) = cat_col_indices.get(&headers[i]) {
                // Handle categorical columns with one-hot encoding
                let encoded = match &headers[*cat_index].as_str() {
                    &"Sex" => match field {
                        "male" => vec![1.0, 0.0],   // Male
                        "female" => vec![0.0, 1.0], // Female
                        _ => vec![0.0, 0.0],        // Unknown
                    },
                    &"Embarked" => match field {
                        "C" => vec![1.0, 0.0, 0.0], // Cherbourg
                        "Q" => vec![0.0, 1.0, 0.0], // Queenstown
                        "S" => vec![0.0, 0.0, 1.0], // Southampton
                        _ => vec![0.0, 0.0, 0.0],   // Unknown
                    },
                    _ => vec![0.0], // Just in case
                };
                one_hot_data.push(encoded);
            } else {
                // Handle numerical columns
                row[i] = field.parse::<f64>().unwrap_or(f64::NAN);
            }
        }

        records.push(row);
    }

    // Flatten one-hot encoded columns and append to records
    let one_hot_flattened: Vec<f64> = one_hot_data.into_iter().flatten().collect();
    let num_rows = records.len();
    let num_cat_cols = one_hot_flattened.len() / num_rows;
    let mut expanded_records = records
        .into_iter()
        .map(|mut row| {
            row.extend(vec![0.0; num_cat_cols]);
            row
        })
        .collect::<Vec<Vec<f64>>>();

    for (i, record) in expanded_records.iter_mut().enumerate() {
        let cat_data_start = i * num_cat_cols;
        record.splice(
            headers.len()..headers.len() + num_cat_cols,
            one_hot_flattened[cat_data_start..cat_data_start + num_cat_cols].to_vec(),
        );
    }

    // Create an ndarray from the CSV records
    let flat_data: Vec<f64> = expanded_records.into_iter().flatten().collect();
    let array = Array2::from_shape_vec((num_rows, headers.len() + num_cat_cols), flat_data)?;

    Ok((headers, array))
}

// Function to print the top rows of the data
fn print_top_rows(data: &Array2<f64>, num_rows: usize) {
    println!("First {} rows of the data:", num_rows);
    for row in data.rows().into_iter().take(num_rows) {
        println!("{:?}", row);
    }
}

// Function to preprocess data: remove rows with NaN and normalize
fn preprocess_data(data: &mut Array2<f64>, numerical_columns: &[usize]) {
    // Remove rows with NaN values in numerical columns
    let mask: Vec<bool> = data
        .rows()
        .into_iter()
        .map(|row| numerical_columns.iter().all(|&col| !row[col].is_nan()))
        .collect();

    let filtered_data: Array2<f64> = data.select(
        Axis(0),
        &mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
            .collect::<Vec<usize>>(),
    );

    // Normalize only the specified numerical columns
    let mut normalized_data = filtered_data.to_owned();

    for &col in numerical_columns {
        // Extract the column
        let column = filtered_data.column(col);
        let mean = column.mean().unwrap_or(0.0);
        let std_dev = column.std(0.0);

        // Normalize the column
        for elem in normalized_data.column_mut(col) {
            *elem = (*elem - mean) / std_dev;
        }
    }

    *data = normalized_data;

    println!("Data has been preprocessed: NaN values removed and features normalized.");
}

// Function to split data into training and testing sets
fn split_data(data: &Array2<f64>, train_size: f64) -> (Array2<f64>, Array2<f64>) {
    let total_rows = data.shape()[0];
    let split_index = (total_rows as f64 * train_size).round() as usize;

    let shuffled_indices: Vec<usize> = (0..total_rows)
        .collect::<Vec<usize>>()
        .choose_multiple(&mut rand::thread_rng(), total_rows)
        .cloned()
        .collect();

    let train_data = data.select(Axis(0), &shuffled_indices[0..split_index]);
    let test_data = data.select(Axis(0), &shuffled_indices[split_index..]);

    (train_data, test_data)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Step 1: Download Titanic dataset
    let raw_data = download_titanic_data()?;

    // Step 2: Load Titanic data into ndarray
    let (_headers, titanic_data) = load_titanic_to_ndarray(&raw_data)?;
    let mut titanic_data = titanic_data.select(Axis(1), &[1, 2, 4, 5, 6, 7, 9, 11]); // Example: selecting a few numerical features

    // Step 3: Preprocess the data
    preprocess_data(&mut titanic_data, &[3, 4, 5, 6]);

    // Step 4: Define features and labels for classification (survival prediction)
    let labels = titanic_data.column(0).to_owned(); // Assuming the 'Survived' column is at index 1
    let labels = labels.clone().into_shape((labels.len(), 1))?;
    let features = titanic_data.select(Axis(1), &[1, 2, 3, 4, 5, 6, 7]); // Example: selecting a few numerical features

    // Step 5: Split data into training and testing sets
    let (train_features, test_features) = split_data(&features, 0.8);
    let (train_labels, test_labels) =
        split_data(&labels.clone().into_shape((labels.len(), 1))?, 0.8); // Cloning labels to avoid moving

    let train_labels = train_labels.iter().cloned().collect::<Array1<f64>>();
    let test_labels = test_labels.iter().cloned().collect::<Array1<f64>>();
    println!("Data has been split into training and testing datasets.");

    // Step 6: Show top rows
    print_top_rows(&train_features, 10);

    // Step 7: Evaluate the quality of the data
    println!("Checking for NaN values in training data...");
    let num_nan_train: usize = train_features.iter().filter(|&&x| x.is_nan()).count();
    let num_nan_test: usize = test_features.iter().filter(|&&x| x.is_nan()).count();

    println!("Number of NaN values in training data: {}", num_nan_train);
    println!("Number of NaN values in test data: {}", num_nan_test);

  	Ok(())
}