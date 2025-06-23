use common::{ReconstructionRequest, ReconstructionResult, ServerStatus};
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;
use chrono::Utc;

const SERVER_URL: &str = "http://127.0.0.1:3000";

#[tokio::main]
async fn main() {
    env_logger::init();
    let client = reqwest::Client::new();

    let monitor_handle = tokio::spawn(monitor_server_performance(client.clone()));
    let reconstruction_handle = tokio::spawn(run_reconstruction_client(client.clone()));

    let _ = tokio::try_join!(monitor_handle, reconstruction_handle);
}

/// Cliente agora apenas envia uma requisição para iniciar a reconstrução no servidor.
async fn run_reconstruction_client(client: reqwest::Client) {
    println!("\n[Cliente] Pressione Enter para enviar uma requisição de reconstrução para o modelo 30x30...");
    let mut buffer = String::new();
    std::io::stdin().read_line(&mut buffer).unwrap();

    let request = ReconstructionRequest {
        user_id: Uuid::new_v4(),
        algorithm_id: "CGNR".to_string(),
        model_id: "30x30".to_string(),
    };

    println!("[Cliente] Enviando requisição para o modelo {}...", request.model_id);

    match client
        .post(format!("{}/reconstruct", SERVER_URL))
        .json(&request)
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                if let Ok(result) = response.json::<ReconstructionResult>().await {
                    println!("[Cliente] Reconstrução concluída com sucesso!");
                    println!("  - Imagem salva no servidor como: img_{}_{}.png", result.user_id, result.end_time.timestamp());
                    println!("  - Iterações executadas: {}", result.iterations);
                    println!("  - Tempo de processamento: {} ms", result.reconstruction_time_ms);
                } else {
                    eprintln!("[Cliente] Erro ao decodificar a resposta do servidor.");
                }
            } else {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                eprintln!("[Cliente] Servidor respondeu com erro: {} - {}", status, text);
            }
        }
        Err(e) => eprintln!("[Cliente] Falha ao conectar com o servidor: {}", e),
    }
}

async fn monitor_server_performance(client: reqwest::Client) {
    println!("\n--- Relatório de Desempenho do Servidor ---");
    println!("{:<25} {:<15} {:<20}", "Horário", "CPU (%)", "Memória (MB)");
    println!("{:-<60}", "");

    for _ in 0..60 { // Monitora por 3 minutos
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
            Err(_e) => {}
        }
        sleep(Duration::from_secs(3)).await;
    }
}