use common::ReconstructionResult;
use ndarray::{Array1, Array2};
use std::error::Error;
use chrono::Utc;
use uuid::Uuid;

/// Lê um vetor de um arquivo CSV (uma coluna de números).
pub fn read_g_vector_from_csv(file_path: &str) -> Result<Array1<f64>, Box<dyn Error>> {
    println!("[Servidor] Lendo vetor 'g' de: {}", file_path);
    let mut reader = csv::ReaderBuilder::new().has_headers(false).from_path(file_path)?;
    let mut data = Vec::new();
    for result in reader.records() {
        let record = result?;
        let value: f64 = record[0].trim().parse()?;
        data.push(value);
    }
    Ok(Array1::from(data))
}

/// Lê uma matriz de um arquivo CSV.
pub fn read_h_matrix_from_csv(file_path: &str, s_samples: usize, n_pixels: usize) -> Result<Array2<f64>, Box<dyn Error>> {
    println!("[Servidor] Lendo matriz 'H' de: {}", file_path);
    let mut reader = csv::ReaderBuilder::new().has_headers(false).from_path(file_path)?;
    let mut flat_data = Vec::with_capacity(s_samples * n_pixels);
    
    for result in reader.records() {
        let record = result?;
        for field in record.iter() {
            let value: f64 = field.trim().parse()?;
            flat_data.push(value);
        }
    }

    if flat_data.len() != s_samples * n_pixels {
        return Err(format!("Erro de dimensão: O arquivo {} contém {} elementos, mas eram esperados {}.", 
            file_path, flat_data.len(), s_samples * n_pixels).into());
    }

    Array2::from_shape_vec((s_samples, n_pixels), flat_data).map_err(|e| e.into())
}

/// Executa o algoritmo CGNR com saídas de depuração.
pub fn execute_cgnr(
    algorithm_id: &str,
    user_id: Uuid,
    h: &Array2<f64>,
    g_signal: &Array1<f64>,
) -> ReconstructionResult {
    let start_time = Utc::now();
    let s = h.shape()[0];
    let n = h.shape()[1];

    // --- [DEBUG] Inspecionar dados de entrada ---
    let g_norm = g_signal.dot(g_signal).sqrt();
    println!("[DEBUG] Norma do vetor 'g' de entrada: {:.4e}", g_norm);

    let mut g = g_signal.clone();
    for l in 0..s {
        let gamma_l = (100.0 + 0.05 * (l as f64).powi(2)).sqrt();
        g[l] *= gamma_l;
    }

    let mut f = Array1::<f64>::zeros(n);
    let mut r = g;
    let mut z = h.t().dot(&r);
    let mut p = z.clone();
    let mut z_t_z = z.dot(&z);

    println!("[DEBUG] Valor inicial de ||z||^2 (z_t_z): {:.4e}", z_t_z);

    let max_iterations = 1000;
    let mut i = 0;
    let convergence_threshold = 1e-4; 

    for iteration_count in 0..max_iterations {
        i = iteration_count;
        let w = h.dot(&p);
        let w_t_w = w.dot(&w);

        if w_t_w.abs() < 1e-20 {
            println!("[DEBUG] Iteração {}: CONDIÇÃO DE PARADA! w_t_w (||w||^2) é quase zero ({:.4e}).", i, w_t_w);
            break;
        }

        let alpha = z_t_z / w_t_w;
        f = f + &(&p * alpha);
        r = r - &(&w * alpha);
        let z_next = h.t().dot(&r);
        
        let z_t_z_next = z_next.dot(&z_next);

        // --- [DEBUG] Imprimir estado da iteração ---
        println!(
            "[DEBUG] Iteração {}: alpha={:.4e}, ||z_i||^2={:.4e}, ||z_{{i+1}}||^2={:.4e}, ||f||={:.4e}",
            i, alpha, z_t_z, z_t_z_next, f.dot(&f).sqrt()
        );
        
        if z_t_z_next < convergence_threshold {
            println!("[Servidor] CGNR convergiu na iteração {}", i + 1);
            break;
        }

        if z_t_z.abs() < 1e-20 {
             println!("[DEBUG] Iteração {}: CONDIÇÃO DE PARADA! z_t_z (||z||^2) é quase zero ({:.4e}). Impossível calcular beta.", i, z_t_z);
             break;
        }
        let beta = z_t_z_next / z_t_z;
        p = &z_next + &(&p * beta);
        z_t_z = z_t_z_next;
    }

    let end_time = Utc::now();
    let reconstruction_time_ms = end_time.signed_duration_since(start_time).num_milliseconds();
    
    // --- [DEBUG] Inspecionar vetor 'f' final ---
    let f_min = f.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let f_max = f.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    println!("[DEBUG] Vetor 'f' final: min={:.4e}, max={:.4e}, range={:.4e}", f_min, f_max, f_max - f_min);

    ReconstructionResult {
        user_id,
        algorithm_id: algorithm_id.to_string(),
        start_time,
        end_time,
        reconstruction_time_ms,
        image_pixels: ((n as f64).sqrt() as usize, (n as f64).sqrt() as usize),
        iterations: i + 1,
        f,
    }
}

/// Salva a imagem, aplicando a inversão vertical.
pub fn save_image(result: &ReconstructionResult) -> Result<(), Box<dyn std::error::Error>> {
    let (height, width) = result.image_pixels;
    if height * width != result.f.len() {
        return Err("Dimensões da imagem não correspondem ao tamanho do vetor 'f'".into());
    }

    // --- INÍCIO DA LÓGICA DE INVERSÃO VERTICAL ---
    println!("[Servidor] Invertendo a imagem verticalmente...");

    let mut flipped_f = Vec::with_capacity(result.f.len());
    
    // DEPOIS: Primeiro obtemos um slice com .as_slice().unwrap() e DEPOIS chamamos .chunks()
    for row_chunk in result.f.as_slice().unwrap().chunks(width).rev() {
    // ANTES: for row_chunk in result.f.chunks(width).rev() {
        flipped_f.extend_from_slice(row_chunk);
    }
    
    // As dimensões da imagem (altura e largura) não mudam em uma inversão.
    // --- FIM DA LÓGICA DE INVERSÃO ---

    // O resto da função agora usa os dados invertidos ('flipped_f').
    let f_min = flipped_f.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let f_max = flipped_f.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = f_max - f_min;

    let image_buffer: Vec<u8> = if range.abs() < 1e-9 {
        vec![0; flipped_f.len()]
    } else {
        flipped_f.iter().map(|&val| (((val - f_min) / range) * 255.0) as u8).collect()
    };
    
    let file_name = format!("img_{}_{}.png", result.user_id, result.end_time.timestamp());

    image::save_buffer(&file_name, &image_buffer, width as u32, height as u32, image::ColorType::L8)?;
    
    println!("[Servidor] Imagem invertida salva como: {}", file_name);
    Ok(())
}