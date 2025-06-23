use chrono::Utc;
use common::{ReconstructionRequest, ReconstructionResult, ServerStatus};
use ndarray::Array;
use ndarray_rand::{rand_distr::Normal, RandomExt};
use rand::{thread_rng, Rng};
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

const SERVER_URL: &str = "http://127.0.0.1:3000";

#[tokio::main]
async fn main() {
    env_logger::init();
    let client = reqwest::Client::new();

    let monitor_handle = tokio::spawn(monitor_server_performance(client.clone()));
    let reconstruction_handle = tokio::spawn(run_reconstruction_client(client.clone()));

    let _ = tokio::try_join!(monitor_handle, reconstruction_handle);
}

async fn run_reconstruction_client(client: reqwest::Client) {
    let mut report_log: Vec<(String, usize, i64)> = Vec::new();

    for i in 0..5 {
        let user_id = Uuid::new_v4();
        println!("\n[Cliente] Preparando requisição {} para o usuário {}", i + 1, user_id);

        let image_dims = (64, 64);
        let model_id_str = format!("{}x{}", image_dims.0, image_dims.1);
        let n_pixels = image_dims.0 * image_dims.1;
        let s_samples = (n_pixels as f64 * 1.5) as usize;

        // O cliente ainda gera H e f para SIMULAR o sinal g resultante
        let h_matrix = Array::random((s_samples, n_pixels), Normal::new(0.0, 1.0).unwrap());
        let true_f = Array::random(n_pixels, Normal::new(10.0, 2.0).unwrap());
        let g_vector = h_matrix.dot(&true_f);

        // A requisição agora é pequena: não inclui mais a matriz H
        let request = ReconstructionRequest {
            user_id,
            // DEPOIS: Peça pelo algoritmo CGNR
            algorithm_id: "CGNR".to_string(),
            // ANTES: algorithm_id: "CGNE".to_string(),
            model_id: model_id_str,
            g: g_vector,
        };

        println!("[Cliente] Enviando g[{}] para o modelo {} usando {}", s_samples, request.model_id, request.algorithm_id);


        match client
            .post(format!("{}/reconstruct", SERVER_URL))
            .json(&request)
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    match response.json::<ReconstructionResult>().await {
                        Ok(result) => {
                            println!("[Cliente] Relatório da Imagem Recebido:");
                            println!("  - Usuário: {}", result.user_id);
                            println!("  - Iterações: {}", result.iterations);
                            println!("  - Tempo de Reconstrução: {} ms", result.reconstruction_time_ms);
                            report_log.push((result.user_id.to_string(), result.iterations, result.reconstruction_time_ms));
                        }
                        Err(e) => eprintln!("[Cliente] Erro ao decodificar a resposta JSON: {}", e),
                    }
                } else {
                    let status = response.status();
                    let text = response.text().await.unwrap_or_default();
                    eprintln!("[Cliente] Servidor respondeu com erro: {} - {}", status, text);
                }
            }
            Err(e) => eprintln!("[Cliente] Falha ao enviar requisição: {}", e),
        }

        let sleep_duration = thread_rng().gen_range(1..=5);
        sleep(Duration::from_secs(sleep_duration)).await;
    }

    println!("\n--- Relatório Final de Imagens Reconstruídas ---");
    println!("{:<38} {:<15} {:<20}", "ID do Usuário", "Iterações", "Tempo de Recon. (ms)");
    println!("{:-<75}", "");
    for (id, iters, time) in report_log {
        println!("{:<38} {:<15} {:<20}", id, iters, time);
    }
}

async fn monitor_server_performance(client: reqwest::Client) {
    println!("\n--- Relatório de Desempenho do Servidor ---");
    println!("{:<25} {:<15} {:<20}", "Horário", "CPU (%)", "Memória (MB)");
    println!("{:-<60}", "");

    for _ in 0..20 { // Monitorar por mais tempo
        match client.get(format!("{}/status", SERVER_URL)).send().await {
            Ok(response) => {
                if let Ok(status) = response.json::<ServerStatus>().await {
                    let now = Utc::now().format("%H:%M:%S.%3f");
                    println!(
                        "{:<25} {:<15.2} {} / {}",
                        now, status.cpu_usage, status.memory_usage_mb, status.total_memory_mb
                    );
                }
            }
            Err(_e) => {}
        }
        sleep(Duration::from_secs(3)).await;
    }
}