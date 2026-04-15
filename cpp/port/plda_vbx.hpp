// SPDX-License-Identifier: MIT
// PLDA transforms (``vbx_setup``) + ``VBx`` / ``cluster_vbx`` (``vbx.py``).

#pragma once

#include <Eigen/Dense>

#include <string>
#include <vector>

namespace pyannote::plda_vbx {

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct PldaModel {
  int lda_dimension = 128;
  Eigen::VectorXd mean1;
  Eigen::VectorXd mean2;
  RowMatrixXd lda;  // NumPy row-major ``(embedding_dim, lda_out)`` from ``xvec_transform.npz``.
  Eigen::VectorXd plda_mu;
  Eigen::MatrixXd plda_tr;
  /// Between-class diagonal in PLDA latent space (``PLDA.phi`` / ``_plda_psi`` head).
  Eigen::VectorXd phi_between;

  void load(const std::string& xvec_transform_npz, const std::string& plda_npz, int lda_dim);
  [[nodiscard]] Eigen::MatrixXd xvec_tf(const Eigen::MatrixXd& embeddings) const;
  [[nodiscard]] Eigen::MatrixXd plda_tf(const Eigen::MatrixXd& x0, int lda_dim) const;
  [[nodiscard]] Eigen::MatrixXd operator()(const Eigen::MatrixXd& embeddings) const;
};

void softmax_rows(const Eigen::MatrixXd& logits, Eigen::MatrixXd& out);

/// ``cluster_vbx`` from ``vbx.py`` (``return_model=True`` path).
/// If ``elbo_epsilon < 0``, ELBO early stopping is disabled (useful for deterministic parity tests).
/// When non-null, ``trace_gamma`` / ``trace_pi`` receive a snapshot after each iteration (post gamma/pi update).
/// ``out_vbx_iters`` / ``out_last_elbo_delta`` (optional) support ``PYANNOTE_CPP_PARITY`` light logging.
void cluster_vbx(
    const std::vector<int>& ahc_init,
    const Eigen::MatrixXd& fea,
    const Eigen::VectorXd& Phi,
    double Fa,
    double Fb,
    int max_iters,
    double init_smoothing,
    Eigen::MatrixXd& gamma,
    Eigen::VectorXd& pi,
    double elbo_epsilon = 1e-4,
    std::vector<Eigen::MatrixXd>* trace_gamma = nullptr,
    std::vector<Eigen::VectorXd>* trace_pi = nullptr,
    int* out_vbx_iters = nullptr,
    double* out_last_elbo_delta = nullptr);

}  // namespace pyannote::plda_vbx
