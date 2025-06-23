mod reconstruction;

use axum::{extract::State, http::StatusCode, response::Json, routing::{get, post}, Router};
use common::{ReconstructionRequest, ReconstructionResult, ServerStatus};
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

/// Handler que recebe a requisição leve e carrega os dados do disco.
async fn handle_reconstruction(
    Json(payload): Json<ReconstructionRequest>,
) -> (StatusCode, Json<ReconstructionResult>) {
    println!("[Servidor] Recebida requisição do usuário {} para o modelo {}", payload.user_id, payload.model_id);

    // Define os arquivos com base no model_id
    let h_file = format!("H-{}.csv", payload.model_id);
    let g_file = format!("g-{}-1.csv", payload.model_id); // Assumindo o sufixo -1
    //let g_file = format!("g-{}-2.csv", payload.model_id); // Assumindo o sufixo -2
    //let g_file = format!("A-{}-1.csv", payload.model_id); // Assumindo o sufixo -1

    // Carrega o vetor g para saber o número de amostras S
    let g_vector = match reconstruction::read_g_vector_from_csv(&g_file) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("[Servidor] ERRO: Falha ao carregar o arquivo g ({}): {}", g_file, e);
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(ReconstructionResult::default()));
        }
    };
    
    let s_samples = g_vector.len(); // S é definido pelo tamanho de g
    let n_pixels = 900; // N = 30x30 = 900

    // Carrega a matriz H com as dimensões corretas
    let h_matrix = match reconstruction::read_h_matrix_from_csv(&h_file, s_samples, n_pixels) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("[Servidor] ERRO: Falha ao carregar o arquivo H ({}): {}", h_file, e);
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(ReconstructionResult::default()));
        }
    };

    // Clona os dados necessários para o thread de processamento
    let user_id = payload.user_id;
    let algorithm_id = payload.algorithm_id;

    let result = tokio::task::spawn_blocking(move || {
        println!("[Servidor] Executando algoritmo {}...", algorithm_id.to_uppercase());
        // Aqui você poderia ter um `if` para escolher o algoritmo se quisesse
        reconstruction::execute_cgnr(&algorithm_id, user_id, &h_matrix, &g_vector)
    }).await.unwrap();

    if let Err(e) = reconstruction::save_image(&result) {
        eprintln!("[Servidor] Erro ao salvar imagem: {}", e);
    }

    (StatusCode::OK, Json(result))
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