mod reconstruction;

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use common::{ReconstructionRequest, ReconstructionResult, ServerStatus}; // ReconstructionResult já importa o trait Default
use std::sync::{Arc, Mutex};
use sysinfo::System;
use tokio::net::TcpListener;

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
        .with_state(shared_state);

    let listener = TcpListener::bind("127.0.0.1:3000").await.unwrap();
    println!("[Servidor] Ouvindo em http://127.0.0.1:3000");
    axum::serve(listener, app).await.unwrap();
}

/// Handler principal que agora orquestra o carregamento do modelo H.
async fn handle_reconstruction(
    Json(payload): Json<ReconstructionRequest>,
) -> (StatusCode, Json<ReconstructionResult>) {
    println!("[Servidor] Recebida requisição do usuário {} para o modelo {}", payload.user_id, payload.model_id);

    let dims: Vec<usize> = match payload.model_id.split('x').map(|s| s.parse()).collect() {
        Ok(d) => d,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(ReconstructionResult::default())),
    };
    if dims.len() != 2 {
        return (StatusCode::BAD_REQUEST, Json(ReconstructionResult::default()));
    }
    let (height, width) = (dims[0], dims[1]);
    let n_pixels = height * width;
    let s_samples = (n_pixels as f64 * 1.5) as usize;

    let h_matrix = reconstruction::get_h_matrix(&payload.model_id, s_samples, n_pixels);

    // DEPOIS: Escolhe qual algoritmo executar com base no pedido do cliente
    let result = tokio::task::spawn_blocking(move || {
        if payload.algorithm_id.to_uppercase() == "CGNR" {
            println!("[Servidor] Executando algoritmo CGNR...");
            reconstruction::execute_cgnr(&payload, &h_matrix)
        } else {
            println!("[Servidor] Executando algoritmo CGNE (padrão)...");
            reconstruction::execute_cgne(&payload, &h_matrix)
        }
    })
    .await
    .unwrap();

    if let Err(e) = reconstruction::save_image(&result) {
        eprintln!("[Servidor] Erro ao salvar imagem: {}", e);
    }

    (StatusCode::OK, Json(result))
}

/// Handler para a rota de status do servidor.
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

// ANTES: O bloco `impl Default for ReconstructionResult` estava aqui e foi removido.