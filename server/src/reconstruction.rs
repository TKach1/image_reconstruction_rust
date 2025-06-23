use common::{ReconstructionRequest, ReconstructionResult};
// DEPOIS: Importações não utilizadas foram removidas para limpar o código.
use ndarray::Array1;
// ANTES: use ndarray::{linalg::Dot, Array, Array1, Array2};
use chrono::Utc;

// Função principal que executa o algoritmo CGNE
pub fn execute_cgne(req: &ReconstructionRequest) -> ReconstructionResult {
    let start_time = Utc::now();

    // Dimensões
    let s = req.h.shape()[0]; // Número de amostras do sinal
    let n = req.h.shape()[1]; // Número de elementos sensores (pixels na imagem final)

    // --- Pré-processamento: Aplicar Ganho de Sinal (γ) ---
    let mut g = req.g.clone();
    for l in 0..s {
        let gamma_l = (100.0 + 0.05 * (l as f64).powi(2)).sqrt();
        g[l] *= gamma_l;
    }

    // --- Inicialização do Algoritmo CGNE ---
    let mut f = Array1::<f64>::zeros(n);
    let mut r = g;
    let mut p = req.h.t().dot(&r);
    let mut r_tr = r.dot(&r);

    // --- Loop de Iterações ---
    let max_iterations = 1000;
    let mut i = 0;
    let convergence_threshold = 1e-4;

    for iteration_count in 0..max_iterations {
        i = iteration_count;
        
        let p_tp = p.dot(&p);
        // Evitar divisão por zero se o vetor p for nulo
        if p_tp.abs() < 1e-12 { break; }
        
        let alpha = r_tr / p_tp;
        f = f + &(&p * alpha);
        r = r - &req.h.dot(&p) * alpha;

        let r_t_r_next = r.dot(&r);
        if r_t_r_next < convergence_threshold {
            println!("Convergência atingida na iteração {}", i + 1);
            break;
        }

        let beta = r_t_r_next / r_tr;
        p = req.h.t().dot(&r) + &(&p * beta);
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
        image_pixels: ( (n as f64).sqrt() as usize, (n as f64).sqrt() as usize ),
        iterations: i + 1,
        f,
    }
}

// Função para salvar a imagem reconstruída em um arquivo
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
        }).collect() // DEPOIS: Ponto e vírgula removido para retornar o valor.
                     // ANTES: }).collect();
    };

    let file_name = format!("img_{}_{}.png", result.user_id, result.end_time.timestamp());
    image::save_buffer(
        &file_name,
        &image_buffer,
        width as u32,
        height as u32,
        image::ColorType::L8,
    )?;

    println!("Imagem salva como: {}", file_name);
    Ok(())
}