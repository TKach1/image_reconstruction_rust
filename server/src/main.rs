mod reconstruction;

use axum::{extract::State, http::StatusCode, response::Json, routing::{get, post}, Router};
use common::{ReconstructionRequest, ReconstructionResult, ServerStatus};
use std::sync::{Arc, Mutex};
use sysinfo::System;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

struct ReconstructionJob {
    request: ReconstructionRequest, // Agora passamos a requisição inteira
    responder: oneshot::Sender<ReconstructionResult>,
}

struct AppState {
    sys: Mutex<System>,
    job_sender: mpsc::Sender<ReconstructionJob>,
}

#[tokio::main]
async fn main() {
    env_logger::init();
    let (job_sender, mut job_receiver) = mpsc::channel::<ReconstructionJob>(10); // Fila com capacidade 10

    tokio::spawn(async move {
        println!("[Worker] Despachante de tarefas iniciado.");
        while let Some(job) = job_receiver.recv().await {
            let request = job.request;
            println!("[Worker] Recebida nova tarefa do usuário {}", request.user_id);

            let h_file = format!("H-{}.csv", request.model_id);

            let s_samples = request.g.len();

            let n_pixels: usize = match request.model_id.as_str() {
                "30x30" => 30 * 30,
                "60x60" => 60 * 60,
                other => {
                    eprintln!("[Worker] ERRO: model_id '{}' não suportado.", other);
                    continue;
                }
            };

            let h_matrix = match reconstruction::read_h_matrix_from_csv(&h_file, s_samples, n_pixels) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("[Worker] ERRO: Falha ao carregar o arquivo H: {}", e);
                    continue;
                }
            };

            let result = tokio::task::spawn_blocking(move || {
                reconstruction::execute_cgnr(&request.algorithm_id, request.user_id, &h_matrix, &request.g)
            }).await.unwrap();

            if let Err(e) = reconstruction::save_image(&result) {
                eprintln!("[Worker] Erro ao salvar imagem: {}", e);
            }

            //RELATORIO FINAL

            if job.responder.send(result).is_err() {
                eprintln!("[Worker] Falha ao enviar resposta. O cliente provavelmente desistiu.");
            }
        }

    });

    let shared_state = Arc::new(AppState {
        sys: Mutex::new(System::new_all()),
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
    let (response_sender, response_receiver) = oneshot::channel();
    let job = ReconstructionJob {
        request: payload,
        responder: response_sender,
    };

    if state.job_sender.try_send(job).is_err() {
        println!("[Servidor] REJEITADO: A fila de tarefas está cheia. Saturação evitada.");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    
    match response_receiver.await {
        Ok(result) => Ok(Json(result)),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
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