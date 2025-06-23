mod reconstruction;

use axum::{
    // DEPOIS: Adicionamos DefaultBodyLimit
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use common::{ReconstructionRequest, ReconstructionResult, ServerStatus};
use std::sync::{Arc, Mutex};
use sysinfo::System;
use tokio::net::TcpListener;

// Estado compartilhado para o servidor, incluindo o monitor do sistema
struct AppState {
    sys: Mutex<System>,
}

#[tokio::main]
async fn main() {
    env_logger::init();
    
    let shared_state = Arc::new(AppState {
        sys: Mutex::new(System::new_all()),
    });

    let app = Router::new()
        .route("/reconstruct", post(handle_reconstruction))
        .route("/status", get(handle_status))
        // DEPOIS: Adicionamos esta camada para aumentar o limite do payload.
        // 250 MB = 250 * 1024 * 1024
        .layer(DefaultBodyLimit::max(250 * 1024 * 1024))
        .with_state(shared_state);

    let listener = TcpListener::bind("127.0.0.1:3000").await.unwrap();
    println!("Servidor ouvindo em http://127.0.0.1:3000");
    axum::serve(listener, app).await.unwrap();
}

// Handler para a rota de reconstrução
async fn handle_reconstruction(
    Json(payload): Json<ReconstructionRequest>,
) -> (StatusCode, Json<ReconstructionResult>) {
    println!("Recebida requisição do usuário: {}", payload.user_id);

    // Executa a reconstrução (CPU-bound, poderia ser movido para um tokio::task::spawn_blocking)
    let result = tokio::task::spawn_blocking(move || {
        reconstruction::execute_cgne(&payload)
    }).await.unwrap();

    // Salva a imagem no disco
    if let Err(e) = reconstruction::save_image(&result) {
        eprintln!("Erro ao salvar imagem: {}", e);
    }

    (StatusCode::OK, Json(result))
}

// Handler para a rota de status do servidor
async fn handle_status(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<ServerStatus>) {
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