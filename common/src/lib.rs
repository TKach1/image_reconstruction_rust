use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Dados enviados pelo cliente para iniciar uma reconstrução.
/// O payload é pequeno, contendo apenas o sinal 'g' e um ID do modelo.
#[derive(Serialize, Deserialize, Debug)]
pub struct ReconstructionRequest {
    pub user_id: Uuid,
    pub algorithm_id: String,
    pub model_id: String, // Ex: "64x64"
    pub g: Array1<f64>,   // Vetor de sinal (g)
}

/// Resultado enviado pelo servidor após a reconstrução.
#[derive(Serialize, Deserialize, Debug)]
pub struct ReconstructionResult {
    pub user_id: Uuid,
    pub algorithm_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub reconstruction_time_ms: i64,
    pub image_pixels: (usize, usize), // (altura, largura) da imagem 'f'
    pub iterations: usize,
    pub f: Array1<f64>, // A imagem reconstruída
}

/// Relatório de status do servidor.
#[derive(Serialize, Deserialize, Debug)]
pub struct ServerStatus {
    pub cpu_usage: f32,
    pub memory_usage_mb: u64,
    pub total_memory_mb: u64,
}

// DEPOIS: O bloco de implementação agora vive aqui, no mesmo crate que ReconstructionResult.
impl Default for ReconstructionResult {
    fn default() -> Self {
        Self {
            user_id: Uuid::nil(),
            algorithm_id: String::from("ERROR"),
            start_time: Utc::now(),
            end_time: Utc::now(),
            reconstruction_time_ms: 0,
            image_pixels: (0, 0),
            iterations: 0,
            f: Array1::zeros(0),
        }
    }
}