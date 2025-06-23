use common::{ReconstructionRequest, ReconstructionResult};
use ndarray::{Array1, Array2};
use ndarray_npy::{read_npy, write_npy};
use ndarray_rand::{rand_distr::Normal, RandomExt};
use std::path::Path;
use chrono::Utc;

/// Busca a matriz H. Se o arquivo .npy correspondente ao model_id existir,
/// ele é carregado. Caso contrário, uma nova matriz é gerada e salva para uso futuro.
pub fn get_h_matrix(model_id: &str, s_samples: usize, n_pixels: usize) -> Array2<f64> {
    let file_path = format!("{}.npy", model_id);
    if Path::new(&file_path).exists() {
        println!("[Servidor] Carregando modelo H existente: {}", file_path);
        read_npy(file_path).expect("Falha ao ler arquivo de modelo .npy")
    } else {
        println!("[Servidor] Gerando novo modelo H para salvar em: {}", file_path);
        let h = Array2::random((s_samples, n_pixels), Normal::new(0.0, 1.0).unwrap());
        // Salva para uso futuro
        write_npy(&file_path, &h).expect("Falha ao salvar arquivo de modelo .npy");
        h
    }
}

/// Executa o algoritmo CGNE. Agora recebe a matriz H como um argumento.
pub fn execute_cgne(req: &ReconstructionRequest, h: &Array2<f64>) -> ReconstructionResult {
    let start_time = Utc::now();

    let s = h.shape()[0];
    let n = h.shape()[1];

    let mut g = req.g.clone();
    for l in 0..s {
        let gamma_l = (100.0 + 0.05 * (l as f64).powi(2)).sqrt();
        g[l] *= gamma_l;
    }

    let mut f = Array1::<f64>::zeros(n);
    let mut r = g;
    let mut p = h.t().dot(&r);
    let mut r_tr = r.dot(&r);

    let max_iterations = 1000;
    let mut i = 0;
    let convergence_threshold = 1e-4;

    for iteration_count in 0..max_iterations {
        i = iteration_count;

        let p_tp = p.dot(&p);
        if p_tp.abs() < 1e-12 { break; }

        let alpha = r_tr / p_tp;
        f = f + &(&p * alpha);
        r = r - &h.dot(&p) * alpha;

        let r_t_r_next = r.dot(&r);
        if r_t_r_next < convergence_threshold {
            println!("[Servidor] Convergência atingida na iteração {}", i + 1);
            break;
        }

        let beta = r_t_r_next / r_tr;
        p = h.t().dot(&r) + &(&p * beta);
        r_tr = r_t_r_next;
    }

    let end_time = Utc::now();
    let reconstruction_time_ms = end_time.signed_duration_since(start_time).num_milliseconds();

    ReconstructionResult {
        user_id: req.user_id,
        algorithm_id: req.algorithm_id.clone(),
        start_time,
        end_time,
        reconstruction_time_ms,
        image_pixels: ((n as f64).sqrt() as usize, (n as f64).sqrt() as usize),
        iterations: i + 1,
        f,
    }
}

/// Executa o algoritmo CGNR (Conjugate Gradient Normal Residual).
/// É numericamente mais estável que o CGNE.
pub fn execute_cgnr(req: &common::ReconstructionRequest, h: &Array2<f64>) -> common::ReconstructionResult {
    let start_time = Utc::now();

    let s = h.shape()[0];
    let n = h.shape()[1];

    // Aplica o ganho de sinal (γ)
    let mut g = req.g.clone();
    for l in 0..s {
        let gamma_l = (100.0 + 0.05 * (l as f64).powi(2)).sqrt();
        g[l] *= gamma_l;
    }

    // --- Inicialização do Algoritmo CGNR ---
    let mut f = Array1::<f64>::zeros(n);
    
    // r_0 = g - H*f_0  (como f_0 é zero, r_0 = g)
    let mut r = g;

    // z_0 = H^T * r_0
    let mut z = h.t().dot(&r);

    // p_0 = z_0
    let mut p = z.clone();
    
    // z_i^T * z_i
    let mut z_t_z = z.dot(&z);

    // --- Loop de Iterações ---
    let max_iterations = 1000;
    let mut i = 0;
    // O critério de convergência para CGNR é a norma do resíduo do sistema normal, ||z||^2
    let convergence_threshold = 1e-4; 

    for iteration_count in 0..max_iterations {
        i = iteration_count;

        // w_i = H * p_i
        let w = h.dot(&p);

        // α_i = ||z_i||^2 / ||w_i||^2
        let w_t_w = w.dot(&w);
        if w_t_w.abs() < 1e-12 { break; } // Evita divisão por zero
        let alpha = z_t_z / w_t_w;

        // f_{i+1} = f_i + α_i * p_i
        f = f + &(&p * alpha);

        // r_{i+1} = r_i - α_i * w_i
        r = r - &(&w * alpha);
        
        // z_{i+1} = H^T * r_{i+1}
        let z_next = h.t().dot(&r);
        
        // Verificar convergência com a norma de z_{i+1}
        let z_t_z_next = z_next.dot(&z_next);
        if z_t_z_next < convergence_threshold {
            println!("[Servidor] CGNR convergiu na iteração {}", i + 1);
            break;
        }

        // β_i = ||z_{i+1}||^2 / ||z_i||^2
        let beta = z_t_z_next / z_t_z;

        // p_{i+1} = z_{i+1} + β_i * p_i
        p = &z_next + &(&p * beta);

        // Atualiza z_t_z para a próxima iteração
        z_t_z = z_t_z_next;
    }

    let end_time = Utc::now();
    let reconstruction_time_ms = end_time.signed_duration_since(start_time).num_milliseconds();

    common::ReconstructionResult {
        user_id: req.user_id,
        algorithm_id: req.algorithm_id.clone(),
        start_time,
        end_time,
        reconstruction_time_ms,
        image_pixels: ((n as f64).sqrt() as usize, (n as f64).sqrt() as usize),
        iterations: i + 1,
        f,
    }
}

/// Salva a imagem reconstruída em um arquivo PNG.
pub fn save_image(result: &ReconstructionResult) -> Result<(), Box<dyn std::error::Error>> {
    let (height, width) = result.image_pixels;
    if height * width != result.f.len() {
        return Err("Dimensões da imagem não correspondem ao tamanho do vetor 'f'".into());
    }

    let f_min = result.f.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let f_max = result.f.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = f_max - f_min;

    let image_buffer: Vec<u8> = if range.abs() < 1e-9 {
        vec![0; result.f.len()]
    } else {
        result.f.iter().map(|&val| {
            ((val - f_min) / range * 255.0) as u8
        }).collect()
    };

    let file_name = format!("img_{}_{}.png", result.user_id, result.end_time.timestamp());
    image::save_buffer(
        &file_name,
        &image_buffer,
        width as u32,
        height as u32,
        image::ColorType::L8,
    )?;

    println!("[Servidor] Imagem salva como: {}", file_name);
    Ok(())
}