mod reconstruction;

use axum::{extract::State, http::StatusCode, response::Json, routing::{get, post}, Router};
// ALTERADO: Importações adicionais
use chrono::Utc;
use common::{ReconstructionRequest, ReconstructionResult, ServerStatus};
use ndarray::Array1;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Arc, Mutex};
use sysinfo::System;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot, Semaphore};
use uuid::Uuid;

struct ReconstructionJob {
    request: ReconstructionRequest,
    responder: oneshot::Sender<ReconstructionResult>,
}

struct AppState {
    sys: Mutex<System>,
    job_sender: mpsc::UnboundedSender<ReconstructionJob>,
}

// ALTERADO: A função agora usa os campos da nova estrutura ReconstructionResult
fn write_report_entry(result: &ReconstructionResult, image_filename: &str) -> std::io::Result<()> {
    const REPORT_FILE: &str = "reconstruction_report.csv";
    
    let file_exists = std::path::Path::new(REPORT_FILE).exists();
    let is_empty = if file_exists {
        std::fs::metadata(REPORT_FILE)?.len() == 0
    } else {
        true
    };

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(REPORT_FILE)?;

    if is_empty {
        // ALTERADO: Cabeçalho do CSV atualizado
        writeln!(file, "user_id,algorithm_id,start_time,end_time,reconstruction_ms,image_pixels,iterations,image_filename")?;
    }
    
    writeln!(
        file,
        "{},{},{},{},{},\"({},{})\",{},{}",
        result.user_id, // Uuid implementa Display, então .to_string() é chamado implicitamente
        result.algorithm_id,
        result.start_time.to_rfc3339(),
        result.end_time.to_rfc3339(),
        result.reconstruction_time_ms,
        result.image_pixels.0,
        result.image_pixels.1,
        result.iterations,
        image_filename
    )?;

    println!("[Worker] Entrada adicionada ao relatório: {}", REPORT_FILE);
    Ok(())
}


#[tokio::main]
async fn main() {
    env_logger::init();

    let (job_sender, mut job_receiver) = mpsc::unbounded_channel::<ReconstructionJob>();

    const MEMORY_UNIT_MB: u64 = 512;
    const MEMORY_30X30_MB: u64 = 512;
    const MEMORY_60X60_MB: u64 = 1536;

    let mut sys = System::new_all();
    sys.refresh_memory();
    
    let total_ram_mb = sys.total_memory() / 1024 / 1024;
    let total_memory_permits = (total_ram_mb / MEMORY_UNIT_MB).max(1) as usize;

    let semaphore = Arc::new(Semaphore::new(total_memory_permits));
    println!(
        "[Servidor] Memória total: {} MB. Pool de workers configurado para {} unidades de memória (1 unidade = {} MB).",
        total_ram_mb, total_memory_permits, MEMORY_UNIT_MB
    );
    
    tokio::spawn(async move {
        println!("[Worker] Despachante de tarefas iniciado.");
        while let Some(job) = job_receiver.recv().await {
            let semaphore_clone = semaphore.clone();

            tokio::spawn(async move {
                let request = job.request; // Mover o request para dentro do escopo
                
                // ALTERADO: Determinar custo de memória e dimensões da imagem
                let (image_pixels, memory_cost_in_units) = match request.model_id.as_str() {
                    "30x30" => ((30, 30), (MEMORY_30X30_MB / MEMORY_UNIT_MB).max(1) as u32),
                    "60x60" => ((60, 60), (MEMORY_60X60_MB / MEMORY_UNIT_MB).max(1) as u32),
                    other => {
                        eprintln!("[Worker] ERRO: model_id '{}' não suportado.", other);
                        let _ = job.responder.send(ReconstructionResult::new_error(request.user_id, request.algorithm_id));
                        return;
                    }
                };

                println!("[Worker] Job do usuário {} aguardando por {} unidade(s) de memória...", request.user_id, memory_cost_in_units);
                let _permit = semaphore_clone.acquire_many_owned(memory_cost_in_units).await.expect("Falha ao adquirir permit do semáforo");
                println!("[Worker] Memória alocada. Iniciando processamento para o usuário {}.", request.user_id);
                
                let h_file = format!("H-{}.csv", request.model_id);
                let s_samples = request.g.len();
                let n_pixels = image_pixels.0 * image_pixels.1;
                
                let h_matrix = match reconstruction::read_h_matrix_from_csv(&h_file, s_samples, n_pixels) {
                    Ok(h) => h,
                    Err(e) => {
                        eprintln!("[Worker] ERRO: Falha ao carregar o arquivo H: {}", e);
                        let _ = job.responder.send(ReconstructionResult::new_error(request.user_id, request.algorithm_id));
                        return;
                    }
                };
                
                // ALTERADO: A tarefa de reconstrução agora retorna a estrutura completa
                let result = tokio::task::spawn_blocking(move || {
                    // Passar os dados necessários para construir o ReconstructionResult
                    reconstruction::execute_cgnr(
                        &request.algorithm_id,
                        request.user_id,
                        &h_matrix,
                        &Array1::from(request.g), // Converter Vec para Array1
                        image_pixels,
                    )
                }).await.unwrap();
                
                let image_filename = match reconstruction::save_image(&result) {
                    Ok(name) => name,
                    Err(e) => {
                        eprintln!("[Worker] Erro ao salvar imagem: {}", e);
                        String::from("save_failed")
                    }
                };

                if let Err(e) = write_report_entry(&result, &image_filename) {
                    eprintln!("[Worker] ERRO: Falha ao escrever no arquivo de relatório: {}", e);
                }

                if job.responder.send(result).is_err() {
                    eprintln!("[Worker] Falha ao enviar resposta. O cliente provavelmente desistiu.");
                }

                 println!("[Worker] Finalizado job do usuário {}. {} unidade(s) de memória liberada(s).", image_filename, memory_cost_in_units);
            });
        }
    });

    let shared_state = Arc::new(AppState {
        sys: Mutex::new(sys),
        job_sender,
    });
    
    let app = Router::new()
        .route("/reconstruct", post(handle_reconstruction))
        .route("/status", get(handle_status))
        .with_state(shared_state);

    let listener = TcpListener::bind("127.0.0.1:3000").await.unwrap();
    println!("[Servidor] Ouvindo em http://127.0.0.1:3000");
    axum::serve(listener, app).await.unwrap();
}

async fn handle_reconstruction(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ReconstructionRequest>,
) -> Result<Json<ReconstructionResult>, StatusCode> {
    let user_id_for_log = payload.user_id;

    let (response_sender, response_receiver) = oneshot::channel();
    let job = ReconstructionJob {
        request: payload,
        responder: response_sender,
    };

    if state.job_sender.send(job).is_err() {
        eprintln!("[Servidor] ERRO CRÍTICO: O despachante de tarefas não está mais recebendo jobs.");
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }
    
    println!("[Servidor] Requisição do usuário {} enfileirada com sucesso.", user_id_for_log);

    match response_receiver.await {
        Ok(result) => {
            // Usar uma heurística para detectar se o resultado é um erro criado por nós
            if result.iterations == 0 && result.reconstruction_time_ms == 0 && result.f.is_empty() {
                 eprintln!("[Servidor] A tarefa para o usuário {} resultou em um erro no worker.", user_id_for_log);
                 Err(StatusCode::BAD_REQUEST)
            } else {
                 Ok(Json(result))
            }
        },
        Err(_) => {
            eprintln!("[Servidor] A tarefa para o usuário {} falhou (canal de resposta fechado).", user_id_for_log);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        },
    }
}

async fn handle_status(State(state): State<Arc<AppState>>) -> (StatusCode, Json<ServerStatus>) {
    let mut sys = state.sys.lock().unwrap();
    sys.refresh_cpu();
    sys.refresh_memory();
    let status = ServerStatus {
        cpu_usage: sys.global_cpu_info().cpu_usage(),
        memory_usage_mb: sys.used_memory() / 1024 / 1024,
        total_memory_mb: sys.total_memory() / 1024 / 1024,
    };
    (StatusCode::OK, Json(status))
}