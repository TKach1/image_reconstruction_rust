use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Dados enviados pelo cliente para iniciar uma reconstrução
#[derive(Serialize, Deserialize, Debug)]
pub struct ReconstructionRequest {
    pub user_id: Uuid,
    pub algorithm_id: String,
    pub g: Array1<f64>, // Vetor de sinal (g)
    pub h: Array2<f64>, // Matriz de modelo (H)
}

// Resultado enviado pelo servidor após a reconstrução
#[derive(Serialize, Deserialize, Debug)]
pub struct ReconstructionResult {
    // Metadados solicitados
    pub user_id: Uuid,
    pub algorithm_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub reconstruction_time_ms: i64,
    pub image_pixels: (usize, usize), // (altura, largura) da imagem `f`
    pub iterations: usize,

    // A imagem reconstruída
    pub f: Array1<f64>,
}

// Relatório de status do servidor
#[derive(Serialize, Deserialize, Debug)]
pub struct ServerStatus {
    pub cpu_usage: f32, // Percentual
    pub memory_usage_mb: u64,
    pub total_memory_mb: u64,
}