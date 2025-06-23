use chrono::Utc;
use common::{ReconstructionRequest, ReconstructionResult, ServerStatus};
use ndarray::Array;
// DEPOIS: Importamos 'Normal' diretamente de 'rand_distr'
use rand_distr::Normal; 
// ANTES: use ndarray_rand::{RandomExt, FNormal};
use ndarray_rand::RandomExt;
use rand::{thread_rng, Rng};
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

const SERVER_URL: &str = "http://127.0.0.1:3000";

#[tokio::main]
async fn main() {
    env_logger::init();
    let client = reqwest::Client::new();

    // Tarefa para monitorar o desempenho do servidor
    let monitor_handle = tokio::spawn(monitor_server_performance(client.clone()));

    // Tarefa principal para enviar requisições de reconstrução
    let reconstruction_handle = tokio::spawn(run_reconstruction_client(client.clone()));

    // Aguarda ambas as tarefas terminarem
    let _ = tokio::try_join!(monitor_handle, reconstruction_handle);
}

// Simula o envio de sinais em intervalos aleatórios
async fn run_reconstruction_client(client: reqwest::Client) {
    let mut report_log: Vec<(String, usize, i64)> = Vec::new();

    for i in 0..5 { // Enviar 5 requisições de exemplo
        let user_id = Uuid::new_v4();
        println!("\n[Cliente] Preparando requisição {} para o usuário {}", i + 1, user_id);

        // Gerar dados aleatórios
        // O usuário, o ganho de sinal e o modelo da imagem deverão ser definidos aleatoriamente;
        let n_pixels = 64 * 64; // Imagem 64x64
        let s_samples = (n_pixels as f64 * 1.5) as usize; // Mais amostras que pixels

        // DEPOIS: Usamos Normal::new(...)
        let h_matrix = Array::random((s_samples, n_pixels), Normal::new(0.0, 1.0).unwrap());
        let true_f = Array::random(n_pixels, Normal::new(10.0, 2.0).unwrap());
        // ANTES: let h_matrix = Array::random((s_samples, n_pixels), FNormal::new(0.0, 1.0).unwrap());

        let g_vector = h_matrix.dot(&true_f);

        let request = ReconstructionRequest {
            user_id,
            algorithm_id: "CGNE".to_string(),
            g: g_vector,
            h: h_matrix,
        };

        println!("[Cliente] Enviando H[{}x{}] e g[{}]", s_samples, n_pixels, s_samples);

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
                            // Adicionar ao log para o relatório final
                            report_log.push((result.user_id.to_string(), result.iterations, result.reconstruction_time_ms));
                        }
                        Err(e) => eprintln!("[Cliente] Erro ao decodificar a resposta JSON: {}", e),
                    }
                } else {
                    eprintln!("[Cliente] Servidor respondeu com erro: {}", response.status());
                }
            }
            Err(e) => eprintln!("[Cliente] Falha ao enviar requisição: {}", e),
        }

        // Intervalo de tempo aleatório entre 1 e 5 segundos
        let sleep_duration = thread_rng().gen_range(1..=5);
        sleep(Duration::from_secs(sleep_duration)).await;
    }

    // Gerar relatório final de imagens
    println!("\n--- Relatório Final de Imagens Reconstruídas ---");
    println!("{:<38} {:<15} {:<20}", "ID do Usuário", "Iterações", "Tempo de Recon. (ms)");
    println!("{:-<75}", "");
    for (id, iters, time) in report_log {
        println!("{:<38} {:<15} {:<20}", id, iters, time);
    }
}


// Monitora o desempenho do servidor periodicamente
async fn monitor_server_performance(client: reqwest::Client) {
    println!("\n--- Relatório de Desempenho do Servidor ---");
    println!("{:<25} {:<15} {:<20}", "Horário", "CPU (%)", "Memória (MB)");
    println!("{:-<60}", "");

    for _ in 0..10 { // Monitorar por um tempo
        match client.get(format!("{}/status", SERVER_URL)).send().await {
            Ok(response) => {
                if let Ok(status) = response.json::<ServerStatus>().await {
                    let now = Utc::now().format("%H:%M:%S");
                    println!(
                        "{:<25} {:<15.2} {} / {}",
                        now, status.cpu_usage, status.memory_usage_mb, status.total_memory_mb
                    );
                }
            }
            // DEPOIS: Usamos _e para silenciar o aviso de variável não utilizada.
            Err(_e) => { /* Servidor pode não estar pronto, ignora */ }
            // ANTES: Err(e) => ...
        }
        sleep(Duration::from_secs(5)).await;
    }
}