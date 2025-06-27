use common::{ReconstructionRequest, ReconstructionResult, ServerStatus};
use ndarray::Array1;
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;
use chrono::Utc;
use rand::Rng;

const SERVER_URL: &str = "http://127.0.0.1:3000";

/// Função para ler o vetor 'g' de um arquivo CSV.
// DEPOIS: A assinatura de erro foi alterada para um tipo específico para ajudar o compilador.
fn read_g_vector_from_csv(file_path: &str) -> Result<Array1<f64>, String> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(file_path)
        .map_err(|e| e.to_string())?;

    let mut data = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| e.to_string())?;
        // DEPOIS: Adicionamos a anotação de tipo para o erro de parse.
        let value: f64 = record[0].trim().parse()
            .map_err(|e: std::num::ParseFloatError| e.to_string())?;
        data.push(value);
    }
    Ok(Array1::from(data))
}

#[tokio::main]
async fn main() {
    env_logger::init();
    let client = reqwest::Client::new();

    let monitor_handle = tokio::spawn(monitor_server_performance(client.clone()));
    let reconstruction_handle = tokio::spawn(run_reconstruction_loop(client.clone()));

    let _ = tokio::try_join!(monitor_handle, reconstruction_handle);
}

/// Envia sinais para o servidor em um loop, escolhendo um arquivo aleatoriamente.
async fn run_reconstruction_loop(client: reqwest::Client) {
    let mut count = 0;
    
    let signal_files = [
        "g-30x30-1.csv", 
        "g-30x30-2.csv", 
        //"A-30x30-1.csv", 
        "G-60x60-1.csv", 
        "G-60x60-2.csv", 
        //"A-60x60-1.csv"
    ];

    loop {
        count += 1;

        // DEPOIS: O gerador de números aleatórios é criado DENTRO do loop.
        // Isso garante que ele não precise ser 'Send' (seguro para threads).
        let random_index = rand::thread_rng().gen_range(0..signal_files.len());
        let signal_file_to_load = signal_files[random_index];

        println!("\n[Cliente] Iteração #{}: Escolhendo arquivo aleatório -> {}", count, signal_file_to_load);

        let g_vector = match read_g_vector_from_csv(signal_file_to_load) {
            Ok(g) => g,
            Err(error_message) => {
                eprintln!("[Cliente] ERRO: Não foi possível ler o arquivo '{}'. Pulando para a próxima iteração. Detalhes: {}", signal_file_to_load, error_message);
                sleep(Duration::from_secs(5)).await;
                continue; 
            }
        };

        // Inferir model_id com base no nome do arquivo CSV
        let model_id = if signal_file_to_load.contains("60x60") {
            "60x60"
        } else {
            "30x30"
        };

        let request = ReconstructionRequest {
            user_id: Uuid::new_v4(),
            algorithm_id: "CGNR".to_string(),
            model_id: model_id.to_string(),
            g: g_vector,
        };

        println!("[Cliente] Enviando requisição #{} para o modelo {}...", count, request.model_id);

        match client
            .post(format!("{}/reconstruct", SERVER_URL))
            .json(&request)
            .send()
            .await
        {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    println!("[Cliente] Requisição #{} processada com sucesso!", count);
                } else {
                    let text = response.text().await.unwrap_or_default();
                    eprintln!("[Cliente] Servidor respondeu à requisição #{} com erro: {} - {}", count, status, text);
                }
            }
            Err(e) => eprintln!("[Cliente] Falha ao enviar a requisição #{}: {}", count, e),
        }
        
        let sleep_time = rand::thread_rng().gen_range(2..=10);
        println!("[Cliente] Aguardando {} segundos para a próxima requisição...", sleep_time);
        sleep(Duration::from_secs(sleep_time)).await;
    }
}

async fn monitor_server_performance(client: reqwest::Client) {
    println!("\n--- Relatório de Desempenho do Servidor ---");
    println!("{:<25} {:<15} {:<20}", "Horário", "CPU (%)", "Memória (MB)");
    println!("{:-<60}", "");

    for _ in 0..60 {
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